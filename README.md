# multistream
# Central CCTV Multi-Camera Video Intelligence Server

Central AI intelligence and indexing server for multi-camera CCTV networks (CAM-01 Main Gate, CAM-02 Parking Area, CAM-03 Exit Gate).

## Architecture

```
CCTV CAM-01 (Laptop / Webcam) ──┐
CCTV CAM-02 (Laptop / Webcam) ──┼── LAN / Wi-Fi (HTTP) ──► THIS CENTRAL SERVER (0.0.0.0:8000)
CCTV CAM-03 (Laptop / Webcam) ──┘                              │
                                                               ├─ FastAPI Ingestion & Heartbeat
                                                               ├─ 1 FPS Downsampler & Ring Buffer
                                                               ├─ SQLite Metadata & Alias Memory
                                                               ├─ Open-Vocabulary Detection (OWL-ViT)
                                                               ├─ Visual Embeddings (CLIP + FAISS)
                                                               ├─ Hybrid Grounded Retrieval
                                                               ├─ On-Demand Evidence Video Clips (FFmpeg)
                                                               └─ React + TypeScript Dashboard (Vite)
```

---

## Key Features & Constraints Honored

- **NO API KEYS Required**: 100% self-hosted, local open-source models with deterministic NLP query parsing.
- **Hardware Acceleration**: Automatic CUDA GPU acceleration (tested with NVIDIA RTX 4050 Laptop GPU) with seamless CPU fallback.
- **Storage Optimized**:
  - Downsampled 1 FPS AI indexing
  - 640x360 standard resolution with JPEG quality 80 compression
  - In-memory rolling ring buffer (20s) for zero-disk high-res video clip generation
  - FAISS is the sole in-memory owner of indexed vectors when available; redundant Python-side copies are kept only by the NumPy fallback
  - Live CCTV streams are not recorded; uploaded source videos are removed after processing
  - Storage cleanup utility `scripts/cleanup.py`
- **Uploaded Video Analysis**: Upload common video formats in the dashboard; isolated background processing samples frames at the configured rate and uses the existing open-vocabulary detector. Ask follow-up questions against that upload only, with timestamped frame evidence and the existing grounded-answer n8n notification flow. Upload processing does not write into live-camera tables or the shared vector index.
- **Persistent Camera Alias Memory**:
  - Automatically maps locations like `main gate` -> `CAM-01`, `parking` -> `CAM-02`, `exit gate` -> `CAM-03`
  - Interactive clarification for unknown locations (e.g. `north gate` -> prompts user once and remembers permanently across restarts)
- **Grounded Verification**: Never fabricates answers; requires verified visual detections and embeddings.
- **Low-Latency Live Preview**: Dashboard camera tiles use the existing MJPEG endpoint, sending only new buffered frames instead of polling still images. Sampled AI indexing runs on a shared background worker so model inference does not delay camera frame acknowledgements.
- **Optional Live Analysis / Model**: A separate feature-flagged worker can analyze the latest frame from one selected live camera using the existing detector or YOLO11n with ByteTrack. Its bounded latest-frame input, event table, and non-blocking n8n queue do not replace or stop the existing camera ingest/index/query pipelines.
- **Expanded Natural-Language Vocabulary**: Query descriptions can include broader everyday objects, clothing, colors and color combinations, patterns, size/appearance details, and common actions (for example, "person carrying a bright orange duffel bag" or "red and navy blue striped shirt"). These terms guide query parsing; visual confirmation still depends on the locally available vision model and camera image quality.
- **Grounded Answer Notifications**: Successful evidence-backed conversational queries send the question, answer, camera, timestamp, confidence, relevance, detections, live/historical status, and evidence frame/clip paths to the configured n8n Telegram workflow. Queries without matching evidence do not trigger a notification; webhook failures are logged and do not alter the CCTV answer.

---

## Project Structure

```
server/
├── backend/
│   ├── config.py              # Configuration & settings
│   ├── main.py                # FastAPI server entrypoint
│   ├── db/
│   │   ├── database.py        # SQLite connection manager with WAL mode
│   │   ├── models.py          # Pydantic schemas
│   │   └── repository.py      # Database CRUD operations
│   ├── ingestion/
│   │   ├── camera_registry.py # Tracks CAM-01, CAM-02, CAM-03, FPS & heartbeats
│   │   ├── buffer_manager.py  # Thread-safe rolling frame memory buffer
│   │   ├── frame_sampler.py   # Downsamples incoming streams to 1 FPS
│   │   └── uploaded_video.py  # Isolated uploaded-video sampling and query jobs
│   ├── models_ai/
│   │   ├── interfaces.py      # Abstract interfaces for AI components
│   │   ├── detector.py        # OWL-ViT open-vocabulary visual detector
│   │   ├── embeddings.py      # Local CLIP image & text embedding model
│   │   └── vector_index.py    # FAISS vector index with disk persistence
│   ├── nlp/
│   │   ├── query_parser.py    # Rule-based natural language parser (no cloud API)
│   │   └── alias_resolver.py  # Persistent camera alias memory in SQLite
│   ├── evidence/
│   │   └── clip_generator.py  # On-demand ±5s evidence video clip generator (FFmpeg/OpenCV)
│   └── routers/
│       ├── health.py          # GET /health
│       ├── cameras.py         # Frame ingestion, status & MJPEG streams
│       ├── query.py           # Natural language grounded query handling
│       ├── uploads.py         # Uploaded-video upload, status, query & evidence APIs
│       ├── evidence.py        # Evidence clips API
│       └── simulation.py      # Multi-camera synthetic stream simulator
├── frontend/                  # React + Vite + TypeScript dashboard
├── data/                      # Sampled frames, evidence videos, SQLite DB, FAISS index
├── models/                    # Model weights cache
├── scripts/
│   ├── simulate_cctv.py       # Standalone 3-camera synthetic video generator
│   ├── client_cctv.py         # Lightweight CCTV client for physical laptops
│   └── cleanup.py             # Storage cleanup utility
├── tests/
│   ├── test_server_pipeline.py # Existing CCTV pipeline tests
│   └── test_uploaded_video.py  # Uploaded-video processing and query tests
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start Commands

### 0. Cache the Open-Vocabulary Detector (One Time, If Needed)
The server looks for OWL-ViT weights in the local Hugging Face cache first. If they are not cached, it uses a limited fallback that cannot reliably identify clothing attributes. To download the configured model weights once:
```bash
ALLOW_ONLINE_DOWNLOADS=1 .venv/bin/python -c "from server.backend.models_ai.detector import detector; detector._load_model(); print('OWL-ViT ready:', detector._model is not None)"
```
This downloads model weights only; no API key is required. Restart the server after the download. Later starts use the local cache and do not need internet access.

### 1. Start the Central Server
```bash
# From workspace root
.venv/bin/uvicorn server.backend.main:app --host 0.0.0.0 --port 8000
```
Server will be accessible at:
- **Web UI & Dashboard**: `http://localhost:8000` (or `http://<SERVER_IP>:8000`)
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

Queries inspect the newest indexed samples from each camera (up to five frames per camera from the last two minutes by default), alongside semantic search results. Adjust `LIVE_QUERY_WINDOW_SEC` and `LIVE_QUERY_FRAMES_PER_CAMERA` in the server environment if needed. A result’s `is_live_evidence` field distinguishes recent camera evidence from older indexed footage; camera status and detector mode are also exposed by `/health`.

### Uploaded Video Analysis

Use **Choose video** in the dashboard to upload an MP4, MOV, M4V, AVI, MKV, or WebM file (up to 512 MB). The server returns a processing status while it samples frames in a dedicated, single-worker background queue. Upload work is bounded to one running and one waiting job, skips redundant reads between sample times and exact duplicate samples, and defers open-vocabulary inference until a question is asked. Repeated identical questions reuse cached results. Once processing completes, select **Uploaded video** as the query source and ask a question in the existing chat. Results include the video-relative time and a viewable evidence frame. A successful grounded match uses the existing n8n webhook and Telegram formatter; unmatched queries do not send notifications.

The upload API is separate from live CCTV:

- `POST /api/uploads/videos` — multipart upload using the `video` field
- `GET /api/uploads/{job_id}` — processing status
- `POST /api/uploads/{job_id}/query` — JSON body such as `{"question":"Did anyone use a mobile phone?"}`
- `GET /api/uploads/{job_id}/frames/{frame_name}` — sampled evidence frame

Uploaded jobs are held in server memory and are not restored after a server restart. The original uploaded file is removed after frame processing; sampled evidence frames remain in that upload's separate `data/uploads` directory.

### Optional Real-time Analysis / Model

Set `REALTIME_ANALYSIS=true` to show the separate Analysis / Model controls in the dashboard. The Upload input reuses the existing upload-and-query pipeline. Live input lets you select any registered camera and run either the existing shared detector or YOLO11n + OpenCV/ByteTrack. YOLO weights load lazily on first use and are reused; provide/cache the configured `REALTIME_YOLO_MODEL` weights before selecting that mode if the server cannot download them.

This optional analyzer consumes only the newest in-memory frame at `REALTIME_INFERENCE_FPS`; it does not record live video or add frames to the shared search index. Only a single camera is actively analyzed by this optional section at a time, and changing its camera/model does not restart the server or stop other cameras. Current detections are displayed over the existing MJPEG stream; event history stores compact detection-backed event metadata in SQLite. Person-entered/exited events require a tracked person at a frame boundary, and possible-carrying events require a detected bag/backpack overlapping the person box.

To limit memory use, YOLO loads only when selected, uses `REALTIME_YOLO_IMAGE_SIZE=416` and a maximum of 50 detections per frame, and runs on CPU by default (`REALTIME_DEVICE=cpu`) rather than competing for GPU memory with the existing models. Its model is released after analysis stops; set `REALTIME_DEVICE=cuda` only when GPU memory is available. The existing detector instance is reused, and its inference calls are serialized with optional YOLO inference when both are configured for GPU use.

Enable `N8N_NOTIFICATIONS=true` separately to send important, cooldown-debounced live-analysis events to the configurable `N8N_WEBHOOK_URL`. Each webhook request runs on a bounded background queue with a single attempt; n8n outages are logged and do not block inference. Existing grounded conversational-answer notifications are unaffected by either new feature flag.

The optional controls use:

- `GET /api/analysis/status` — enabled/running state, current detections and notification status
- `POST /api/analysis/start` — JSON body such as `{"camera_id":"CAM-01","model":"yolo11n"}`
- `POST /api/analysis/stop` — stop only this optional analyzer
- `GET /api/analysis/events?camera_id=CAM-01&limit=30` — recent compact event history

Set `REALTIME_INFERENCE_FPS`, `REALTIME_FRAME_WIDTH`, `REALTIME_FRAME_HEIGHT`, and `REALTIME_YOLO_IMAGE_SIZE` to tune analysis load; `REALTIME_EVENT_COOLDOWN_SEC` (20 seconds by default) debounces repeated camera/event/tracking-ID events. The new settings are read from the process environment or the server `.env` file. With both feature flags false, the dashboard hides this section and the existing behavior remains unchanged.

### 2. Connect Physical CCTV Laptops (CAM-01, CAM-02, CAM-03)
On each CCTV laptop running over Wi-Fi / LAN:

**Laptop 1 (Main Gate):**
```bash
python scripts/client_cctv.py --camera-id CAM-01 --camera-name "Main Gate" --server http://<SERVER_IP>:8000 --device 0
```

**Laptop 2 (Parking Area):**
```bash
python scripts/client_cctv.py --camera-id CAM-02 --camera-name "Parking Area" --server http://<SERVER_IP>:8000 --device 0
```

**Laptop 3 (Exit Gate):**
```bash
python scripts/client_cctv.py --camera-id CAM-03 --camera-name "Exit Gate" --server http://<SERVER_IP>:8000 --device 0
```

### 3. Simulation Mode (Test Without Physical Cameras)
You can either click the **"Start Multi-Cam Demo Simulation"** button in the Web UI, or run:
```bash
.venv/bin/python server/scripts/simulate_cctv.py --server http://127.0.0.1:8000 --fps 2.0
```

### 4. Run All Automated Tests
```bash
.venv/bin/pytest server/tests/test_server_pipeline.py -v
.venv/bin/pytest server/tests/test_uploaded_video.py -v
.venv/bin/pytest server/tests/test_realtime_analysis.py -v
```

### 5. Storage Cleanup
```bash
.venv/bin/python server/scripts/cleanup.py --all-evidence
```

---

## Storage & Performance Estimates

| Component | Resolution / Rate | Storage Per Hour | 24-Hour Estimate |
| :--- | :--- | :--- | :--- |
| **Incoming Video Streams** | 640x360 @ 5-15 FPS | 0 MB (kept in RAM ring buffer only) | 0 MB |
| **Sampled Indexed Frames** | 640x360 @ 1 FPS JPEG | ~70 MB / camera | ~1.6 GB / camera |
| **Vector Embeddings (FAISS)**| 512-dim Float32 | ~7.3 MB / camera | ~175 MB total |
| **Evidence Video Clips** | ±5s MP4 H.264 | Generated on-demand (~1.2 MB / clip) | Auto-pruned |
