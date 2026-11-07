"""番茄图片混淆 / 解混淆 AstrBot 插件主入口。

提供三种触发方式：
1. 指令：`/混淆`、`/解混淆`
2. 关键词：发送或引用图片 + 命中关键词（如"打乱"、"还原"等）
3. LLM 工具调用：LLM 在自然语言对话中自动调用本插件的工具

混淆算法基于 Gilbert 空间填充曲线 + 黄金比例偏移，参考 xiaofanqie-source。
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import time
import uuid
from typing import List, Optional

import aiohttp
from PIL import Image

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

try:
    from .obfuscator import deobfuscate, obfuscate, save_jpeg, save_png
except ImportError:  # 兼容直接以模块名加载的情况
    from obfuscator import deobfuscate, obfuscate, save_jpeg, save_png


@register(
    "astrbot_plugin_tomato_obfuscate",
    "Bigesila-B",
    "番茄图片混淆：基于Gilbert空间填充曲线的图片混淆/解混淆插件，支持指令、关键词、LLM智能调用三种触发方式。",
    "1.0.1",
)
class TomatoObfuscatePlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self._tmp_dir = os.path.join(tempfile.gettempdir(), "tomato_obfuscate")
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._processed_keys: set = set()  # 防止同一消息被指令和关键词重复触发

    # ---------- 配置与工具方法 ----------
    def _cfg(self, key: str, default=None):
        config = getattr(self, "config", None) or {}
        return config.get(key, default)

    def _extract_images(self, event: AstrMessageEvent) -> List[Comp.Image]:
        """从消息链中提取所有 Image 段。"""
        return [c for c in event.get_messages() if isinstance(c, Comp.Image)]

    async def _get_quoted_images(self, event: AstrMessageEvent) -> List[Comp.Image]:
        """从引用消息中获取图片（支持 aiocqhttp/napcat 平台）。

        napcat 等 OneBot v11 适配器不会自动把被引用消息合并到消息链，
        需要通过 Reply 段的 message_id 主动调用 get_msg API 获取。
        """
        if event.get_platform_name() != "aiocqhttp":
            return []

        try:
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )
        except ImportError:
            try:
                from astrbot.core.platform.sources.aiocqhttp import AiocqhttpMessageEvent
            except ImportError:
                return []

        if not isinstance(event, AiocqhttpMessageEvent):
            return []

        # 从消息链中找 Reply 段
        reply_id = None
        for comp in event.message_obj.message:
            if getattr(comp, "type", "") == "Reply":
                reply_id = getattr(comp, "id", None)
                break

        if not reply_id:
            return []

        try:
            client = event.bot
            quoted_msg = await client.api.call_action("get_msg", message_id=int(reply_id))
        except Exception as e:
            logger.warning(f"获取引用消息失败: {e}")
            return []

        # 从被引用消息中提取图片
        images: List[Comp.Image] = []
        message_segs = quoted_msg.get("message", []) if isinstance(quoted_msg, dict) else []
        for seg in message_segs:
            if isinstance(seg, dict) and seg.get("type") == "image":
                url = seg.get("data", {}).get("url", "")
                if url:
                    images.append(Comp.Image(file=url))
        return images

    async def _collect_all_images(self, event: AstrMessageEvent) -> List[Comp.Image]:
        """收集消息中的所有图片，包括引用消息中的图片。"""
        images = self._extract_images(event)
        if images:
            return images
        # 当前消息没有图片，尝试从引用消息中获取
        return await self._get_quoted_images(event)

    async def _download_image(self, url: str) -> bytes:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def _load_image(self, comp: Comp.Image) -> Optional[Image.Image]:
        """从 Image 消息段加载 PIL.Image。依次尝试本地路径、base64、URL。"""
        file = (getattr(comp, "file", None) or "").strip()
        url = (getattr(comp, "url", None) or "").strip()
        path = (getattr(comp, "path", None) or "").strip()

        # 1. 本地文件路径
        for candidate in (path, file):
            if candidate and not candidate.startswith("base64://") and os.path.isfile(candidate):
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception as e:
                    logger.warning(f"打开本地图片失败 {candidate}: {e}")

        # 2. base64
        if file.startswith("base64://"):
            try:
                data = base64.b64decode(file[len("base64://"):])
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                logger.warning(f"解析 base64 图片失败: {e}")

        # 3. URL（url 字段或 file 字段是 http(s) 链接）
        for candidate in (url, file):
            if candidate and (candidate.startswith("http://") or candidate.startswith("https://")):
                try:
                    data = await self._download_image(candidate)
                    return Image.open(io.BytesIO(data)).convert("RGB")
                except Exception as e:
                    logger.warning(f"下载图片失败 {candidate}: {e}")

        return None

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        max_pixels = int(self._cfg("max_image_pixels", 2073600))
        w, h = img.size
        if w * h > max_pixels and max_pixels > 0:
            ratio = (max_pixels / (w * h)) ** 0.5
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"图片过大，缩放至 {new_w}x{new_h}")
        return img

    def _save_temp(self, img: Image.Image, use_png: bool = False) -> str:
        """保存临时图片。use_png=True 时用 PNG 无损格式（适合需要解混淆还原的混淆图）。"""
        if use_png:
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(self._tmp_dir, filename)
            save_png(img, path)
        else:
            quality = float(self._cfg("jpeg_quality", 0.95))
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
            path = os.path.join(self._tmp_dir, filename)
            save_jpeg(img, path, quality=quality)
        return path

    def _match_keywords(self, text: str, keywords_cfg: str) -> bool:
        keywords = [k.strip() for k in (keywords_cfg or "").split("|") if k.strip()]
        return any(k in text for k in keywords)

    async def _process(self, event: AstrMessageEvent, mode: str):
        """mode: 'obfuscate' 或 'deobfuscate'。"""
        # 防止同一消息的同一种操作被重复触发（指令与关键词同时命中）
        msg_id = getattr(event.message_obj, "message_id", "") or str(id(event))
        cache_key = f"{msg_id}:{mode}"
        if cache_key in self._processed_keys:
            return
        self._processed_keys.add(cache_key)
        if len(self._processed_keys) > 200:
            # 保留最近 100 个，避免集合无限增长
            self._processed_keys = set(list(self._processed_keys)[-100:])

        images = await self._collect_all_images(event)
        action = "混淆" if mode == "obfuscate" else "解混淆"
        if not images:
            yield event.plain_result(
                f"未检测到图片。请发送或引用一张图片，并附带「{action}」关键词。"
            )
            return

        if len(images) > 1:
            yield event.plain_result(f"检测到 {len(images)} 张图片，将依次{action}。")

        success = 0
        for idx, comp in enumerate(images, 1):
            try:
                img = await self._load_image(comp)
                if img is None:
                    yield event.plain_result(f"第 {idx} 张图片加载失败，跳过。")
                    continue
                img = self._resize_if_needed(img)
                result = obfuscate(img) if mode == "obfuscate" else deobfuscate(img)
                # 混淆图用 PNG（无损），解混淆图用 JPEG
                path = self._save_temp(result, use_png=(mode == "obfuscate"))
                prefix = f"第 {idx} 张" if len(images) > 1 else "图片"
                if mode == "obfuscate":
                    hint = f"（原始尺寸 {img.size[0]}×{img.size[1]}，PNG 无损，请保留此图用于解混淆）"
                else:
                    hint = ""
                yield event.chain_result([
                    Comp.Plain(f"{prefix}{action}完成{hint}：\n"),
                    Comp.Image.fromFileSystem(path),
                ])
                success += 1
            except Exception as e:
                logger.exception(f"处理图片失败: {e}")
                yield event.plain_result(f"第 {idx} 张图片处理失败：{e}")

        if success == 0:
            yield event.plain_result(f"没有图片被成功{action}。")

    # ---------- 方式一：直接指令 ----------
    @filter.command("混淆")
    async def cmd_obfuscate(self, event: AstrMessageEvent):
        """对图片执行混淆操作。"""
        async for r in self._process(event, "obfuscate"):
            yield r

    @filter.command("解混淆")
    async def cmd_deobfuscate(self, event: AstrMessageEvent):
        """对图片执行解混淆操作。"""
        async for r in self._process(event, "deobfuscate"):
            yield r

    # ---------- 方式二：关键词触发（监听所有消息，命中关键词且含图片才处理）----------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        if not text or text.startswith("/"):
            return  # 跳过空消息和指令（指令由上面的 command handler 处理）

        images = await self._collect_all_images(event)
        if not images:
            return  # 没图片不处理，避免误触发

        ob_kw = self._cfg("obfuscate_keywords", "混淆|打乱|加密图片|加密图|打乱图片|混淆图片|搅乱")
        de_kw = self._cfg("deobfuscate_keywords", "解混淆|还原|解密图片|解密图|解开混淆|还原图片|解开")

        # 解混淆关键词优先级更高（避免"解混淆"被"混淆"先命中）
        if self._match_keywords(text, de_kw):
            event.stop_event()
            async for r in self._process(event, "deobfuscate"):
                yield r
            return

        if self._match_keywords(text, ob_kw):
            event.stop_event()
            async for r in self._process(event, "obfuscate"):
                yield r
            return

    # ---------- 方式三：LLM 工具调用 ----------
    @filter.llm_tool(name="tomato_obfuscate_image")
    async def llm_obfuscate(self, event: AstrMessageEvent):
        '''对用户在当前对话中发送或引用的图片执行混淆（打乱像素）操作。

        适用场景：用户发送了一张图片，并表达"混淆图片"、"打乱图片"、"加密图片像素"、"把图片打乱"等意图时调用此工具。
        调用此工具不需要传入图片参数，工具会自动从当前对话上下文中获取用户发送的图片。
        混淆后图片视觉上呈油画/噪点效果，但保持空间相关性，可被解混淆还原。

        Args:
        '''
        if not self._cfg("enable_llm_tool", True):
            yield event.plain_result("图片混淆的 LLM 工具已被禁用。")
            return
        async for r in self._process(event, "obfuscate"):
            yield r

    @filter.llm_tool(name="tomato_deobfuscate_image")
    async def llm_deobfuscate(self, event: AstrMessageEvent):
        '''对用户在当前对话中发送或引用的图片执行解混淆（还原像素）操作。

        适用场景：用户发送了一张已被混淆的图片，并表达"解混淆"、"还原图片"、"解开混淆"、"解密图片"等意图时调用此工具。
        调用此工具不需要传入图片参数，工具会自动从当前对话上下文中获取用户发送的图片。
        仅能还原由本插件混淆的图片；若原图经过其他处理可能无法完美还原。

        Args:
        '''
        if not self._cfg("enable_llm_tool", True):
            yield event.plain_result("图片解混淆的 LLM 工具已被禁用。")
            return
        async for r in self._process(event, "deobfuscate"):
            yield r
