import logging
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

N8N_WEBHOOK_URL = "https://nitharsh.app.n8n.cloud/webhook/video-events"
TELEGRAM_MESSAGE_LIMIT = 4096


def _format_timestamp(value: Any) -> str:
    if not value:
        return "Unknown time"

    timestamp = str(value)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp

    if parsed.tzinfo is None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_telegram_message(payload: Dict[str, Any]) -> str:
    lines = [
        f"📍 {payload.get('camera', 'Unknown camera')} · {_format_timestamp(payload.get('timestamp'))}",
        str(payload.get("answer") or "No answer details available."),
    ]

    question = payload.get("question")
    if question:
        lines.append(f"Q: {question}")

    confidence = payload.get("confidence")
    relevance = payload.get("relevance")
    if isinstance(confidence, (int, float)) and isinstance(relevance, (int, float)):
        lines.append(f"Confidence {confidence:.0%} · Relevance {relevance:.0%}")

    if payload.get("is_live_evidence"):
        lines.append("Live evidence")
    elif isinstance(payload.get("source_age_seconds"), (int, float)):
        age_minutes = payload["source_age_seconds"] / 60
        lines.append(f"Historical · {age_minutes:.1f} min old")

    detections = payload.get("detections")
    if isinstance(detections, list) and detections:
        labels = [
            f"{item['label']} {item['confidence']:.0%}"
            if isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("confidence"), (int, float))
            else str(item.get("label", "object"))
            for item in detections[:5]
            if isinstance(item, dict)
        ]
        if labels:
            more = f" +{len(detections) - 5}" if len(detections) > 5 else ""
            lines.append(f"Detected: {', '.join(labels)}{more}")

    if payload.get("evidence_frame"):
        lines.append(f"Frame: {payload['evidence_frame']}")
    if payload.get("evidence_clip"):
        lines.append(f"Clip: {payload['evidence_clip']}")

    message = "\n".join(lines)
    if len(message) > TELEGRAM_MESSAGE_LIMIT:
        message = message[: TELEGRAM_MESSAGE_LIMIT - 1].rstrip() + "…"
    return message


async def send_answer_notification(payload: Dict[str, Any]) -> None:
    """Post a grounded query answer to n8n without blocking the query response."""
    timeout = httpx.Timeout(5.0, connect=2.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            webhook_payload = {**payload, "text": _format_telegram_message(payload)}
            response = await client.post(N8N_WEBHOOK_URL, json=webhook_payload)
            if response.is_error:
                logger.warning(
                    "n8n webhook rejected grounded answer: HTTP %s; response=%s",
                    response.status_code,
                    response.text[:500],
                )
                return
            logger.info("Grounded answer delivered to n8n webhook (HTTP %s).", response.status_code)
    except httpx.HTTPError as exc:
        logger.warning("Could not deliver grounded answer to n8n webhook: %s", exc)
