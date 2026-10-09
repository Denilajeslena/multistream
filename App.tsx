import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, 
  Send, 
  Play, 
  Video, 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  ShieldAlert, 
  RefreshCw,
  Upload
} from 'lucide-react';

interface CameraInfo {
  camera_id: string;
  camera_name: string;
  status: 'ONLINE' | 'OFFLINE';
  last_seen: number;
  fps: number;
  frame_count: number;
}

interface DetectionBox {
  label: string;
  confidence: number;
  box: number[]; // [x1, y1, x2, y2]
}

interface GroundedResult {
  matched: boolean;
  camera_id?: string;
  camera_name?: string;
  timestamp?: number;
  timestamp_iso?: string;
  frame_id?: string;
  frame_url?: string;
  clip_url?: string;
  confidence: number;
  relevance: number;
  detections: DetectionBox[];
  is_live_evidence?: boolean;
  source_age_seconds?: number;
  timeline?: Array<{
    camera_id: string;
    camera_name: string;
    timestamp: number;
    timestamp_iso?: string;
    label?: string;
    confidence?: number;
  }>;
  explanation: string;
  unresolved_alias_prompt?: string;
  source?: 'uploaded_video';
}

interface UploadedVideoJob {
  job_id: string;
  filename: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  total_frames: number;
  frame_count: number;
  duration_seconds: number;
  error?: string | null;
}

interface RealtimeDetection {
  label: string;
  confidence: number;
  box: number[];
  tracking_id: number | null;
}

interface RealtimeEvent {
  id?: number;
  camera_id: string;
  timestamp: number;
  event_type: string;
  objects: string[];
  confidence: number;
  tracking_ids: number[];
  bounding_box: number[] | null;
}

interface RealtimeAnalysisStatus {
  enabled: boolean;
  active: boolean;
  camera_id: string | null;
  model: 'existing' | 'yolo11n';
  inference_fps: number;
  frame_timestamp: number | null;
  frame_width: number;
  frame_height: number;
  detections: RealtimeDetection[];
  current_events: RealtimeEvent[];
  error: string | null;
  notifications_enabled: boolean;
  notification_status: string;
  last_notification: { timestamp: number | string; event_type: string; status: string } | null;
  cooldown_seconds: number;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'system';
  text: string;
  result?: GroundedResult;
  timestamp: string;
  originalQuery?: string;
}

export const App: React.FC = () => {
  const [cameras, setCameras] = useState<CameraInfo[]>([
    { camera_id: 'CAM-01', camera_name: 'Main Gate', status: 'OFFLINE', last_seen: 0, fps: 0, frame_count: 0 },
    { camera_id: 'CAM-02', camera_name: 'Parking Area', status: 'OFFLINE', last_seen: 0, fps: 0, frame_count: 0 },
    { camera_id: 'CAM-03', camera_name: 'Exit Gate', status: 'OFFLINE', last_seen: 0, fps: 0, frame_count: 0 },
  ]);

  const [simulationRunning, setSimulationRunning] = useState<boolean>(false);
  const [detectorMode, setDetectorMode] = useState<'owlvit' | 'fallback' | 'not_loaded'>('not_loaded');
  const [queryInput, setQueryInput] = useState<string>('');
  const [loadingQuery, setLoadingQuery] = useState<boolean>(false);
  const [selectedEvidence, setSelectedEvidence] = useState<GroundedResult | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-01');
  const [uploadedVideo, setUploadedVideo] = useState<UploadedVideoJob | null>(null);
  const [uploadedVideoResult, setUploadedVideoResult] = useState<GroundedResult | null>(null);
  const [uploadingVideo, setUploadingVideo] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string>('');
  const [querySource, setQuerySource] = useState<'live' | 'uploaded'>('live');
  const [analysisInput, setAnalysisInput] = useState<string>('CAM-01');
  const [analysisModel, setAnalysisModel] = useState<'existing' | 'yolo11n'>('existing');
  const [analysisStatus, setAnalysisStatus] = useState<RealtimeAnalysisStatus | null>(null);
  const [analysisEvents, setAnalysisEvents] = useState<RealtimeEvent[]>([]);
  const [analysisActionError, setAnalysisActionError] = useState('');
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-1',
      sender: 'system',
      text: 'Central CCTV Video Intelligence system initialized. Ready to receive streams from CAM-01, CAM-02, and CAM-03. You can ask natural-language questions about vehicles, people, or events.',
      timestamp: new Date().toLocaleTimeString(),
    }
  ]);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const selectedTimeline = selectedEvidence?.timeline ?? [];
  const uploadJobId = uploadedVideo?.job_id;
  const uploadIsProcessing = uploadedVideo?.status === 'processing';

  // Poll camera status and simulation status
  const fetchStatus = async () => {
    try {
      const camRes = await fetch('/api/cameras');
      if (camRes.ok) {
        const data = await camRes.json();
        setCameras(data);
      }
      const simRes = await fetch('/api/simulation/status');
      if (simRes.ok) {
        const simData = await simRes.json();
        setSimulationRunning(simData.running);
      }
      const healthRes = await fetch('/health');
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setDetectorMode(healthData.vision_models?.detector || 'not_loaded');
      }
    } catch (err) {
      console.error('Status fetch error:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let active = true;
    let interval: number | undefined;
    const refreshAnalysis = async () => {
      try {
        const statusResponse = await fetch('/api/analysis/status');
        if (!statusResponse.ok) return;
        const status: RealtimeAnalysisStatus = await statusResponse.json();
        if (!active) return;
        setAnalysisStatus(status);
        if (status.enabled) {
          const eventsResponse = await fetch('/api/analysis/events?limit=30');
          if (active && eventsResponse.ok) setAnalysisEvents(await eventsResponse.json());
          if (interval === undefined) interval = window.setInterval(refreshAnalysis, 1500);
        }
      } catch (err) {
        console.error('Analysis status fetch error:', err);
      }
    };
    refreshAnalysis();
    return () => {
      active = false;
      if (interval !== undefined) clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!uploadJobId || !uploadIsProcessing) return;
    let active = true;
    const refreshUploadStatus = async () => {
      try {
        const response = await fetch(`/api/uploads/${uploadJobId}`);
        const data = await response.json().catch(() => ({}));
        if (!active) return;
        if (response.ok) {
          setUploadedVideo(data as UploadedVideoJob);
          setUploadError('');
        } else if (response.status === 404) {
          setUploadedVideo((current) => current?.job_id === uploadJobId
            ? { ...current, status: 'failed', error: 'The server no longer has this upload job. Upload the video again.' }
            : current);
        } else {
          setUploadError(data.detail || `Could not check upload progress (HTTP ${response.status}).`);
        }
      } catch (err) {
        console.error('Uploaded video status fetch error:', err);
        if (active) setUploadError('Cannot reach the server to check upload progress. Retrying…');
      }
    };
    const interval = setInterval(refreshUploadStatus, 1500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [uploadJobId, uploadIsProcessing]);

  const toggleSimulation = async () => {
    try {
      if (simulationRunning) {
        await fetch('/api/simulation/stop', { method: 'POST' });
        setSimulationRunning(false);
      } else {
        await fetch('/api/simulation/start?fps=2.0', { method: 'POST' });
        setSimulationRunning(true);
      }
      setTimeout(fetchStatus, 600);
    } catch (err) {
      console.error('Simulation toggle error:', err);
    }
  };

  const handleSendQuery = async (queryText?: string) => {
    const q = (queryText || queryInput).trim();
    if (!q || loadingQuery) return;
    if (querySource === 'uploaded' && uploadedVideo?.status !== 'completed') return;

    const userMsg: ChatMessage = {
      id: `usr-${Date.now()}`,
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQueryInput('');
    setSelectedEvidence(null);
    setLoadingQuery(true);

    try {
      const useUploadedVideo = querySource === 'uploaded' && uploadedVideo?.status === 'completed';
      const resp = await fetch(
        useUploadedVideo ? `/api/uploads/${uploadedVideo.job_id}/query` : '/api/query',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(useUploadedVideo ? { question: q } : { query: q }),
        }
      );

      if (!resp.ok) {
        const error = await resp.json().catch(() => ({}));
        throw new Error(error.detail || `Server returned HTTP ${resp.status}`);
      }

      const result: GroundedResult = await resp.json();

      const sysMsg: ChatMessage = {
        id: `sys-${Date.now()}`,
        sender: 'system',
        text: result.explanation,
        result: result,
        timestamp: new Date().toLocaleTimeString(),
        originalQuery: q,
      };

      setMessages((prev) => [...prev, sysMsg]);

      if (useUploadedVideo) setUploadedVideoResult(result);
      if (result.matched) {
        setSelectedEvidence(result);
      }
    } catch (err: any) {
      const errorMessage = err.message || 'Unknown network error';
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-err-${Date.now()}`,
          sender: 'system',
          text: `Error processing query: ${errorMessage}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
      if (querySource === 'uploaded') setUploadError(`Could not query uploaded video: ${errorMessage}`);
    } finally {
      setLoadingQuery(false);
    }
  };

  const handleVideoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    setUploadingVideo(true);
    setUploadError('');
    setUploadedVideo(null);
    setUploadedVideoResult(null);
    setSelectedEvidence(null);
    if (file.size > 512 * 1024 * 1024) {
      setUploadError('Video exceeds the 512 MB upload limit.');
      setUploadingVideo(false);
      return;
    }
    try {
      const formData = new FormData();
      formData.append('video', file);
      const response = await fetch('/api/uploads/videos', { method: 'POST', body: formData });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Server returned HTTP ${response.status}`);
      }
      const job: UploadedVideoJob = await response.json();
      setUploadedVideo(job);
      setQuerySource('uploaded');
    } catch (err: any) {
      setUploadError(err.message || 'Video upload failed.');
    } finally {
      setUploadingVideo(false);
    }
  };

  const handleAnalysisToggle = async () => {
    setAnalysisActionError('');
    try {
      const response = await fetch(
        analysisStatus?.active ? '/api/analysis/stop' : '/api/analysis/start',
        analysisStatus?.active
          ? { method: 'POST' }
          : {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ camera_id: analysisInput, model: analysisModel }),
            }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Server returned HTTP ${response.status}`);
      setAnalysisStatus(data);
    } catch (err: any) {
      setAnalysisActionError(err.message || 'Could not update real-time analysis.');
    }
  };

  const handleResolveAlias = async (alias: string, cameraId: string, originalQuery?: string) => {
    setLoadingQuery(true);
    try {
      const resp = await fetch('/api/query/alias-resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alias: alias,
          camera_id: cameraId,
          original_query: originalQuery,
        }),
      });

      if (resp.ok) {
        const result: GroundedResult = await resp.json();
        const sysMsg: ChatMessage = {
          id: `sys-alias-${Date.now()}`,
          sender: 'system',
          text: `Mapped location '${alias}' to ${cameraId}. ` + (result.matched ? result.explanation : 'No matching visual evidence found.'),
          result: result,
          timestamp: new Date().toLocaleTimeString(),
        };
        setMessages((prev) => [...prev, sysMsg]);
        if (result.matched) {
          setSelectedEvidence(result);
        }
      }
    } catch (err: any) {
      console.error('Alias resolve error:', err);
    } finally {
      setLoadingQuery(false);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Top Header */}
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ background: '#2563eb', padding: '8px', borderRadius: '8px', display: 'flex' }}>
            <Camera size={22} color="white" />
          </div>
          <div>
            <h1 style={{ fontSize: '17px', fontWeight: '700', letterSpacing: '-0.02em' }}>
              CENTRAL CCTV INTELLIGENCE SYSTEM
            </h1>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Multi-Camera Ingestion • Open-Vocabulary Grounding • Local Embeddings
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Active Cameras Count */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
            <span style={{ color: '#94a3b8' }}>Streams:</span>
            <span style={{ fontWeight: '600', color: cameras.filter(c => c.status === 'ONLINE').length > 0 ? '#34d399' : '#f87171' }}>
              {cameras.filter(c => c.status === 'ONLINE').length} / {cameras.length} Online
            </span>
          </div>

          {/* Simulation Toggle */}
          <button 
            onClick={toggleSimulation}
            className={`btn ${simulationRunning ? 'btn-secondary' : 'btn-accent'}`}
            style={{ border: simulationRunning ? '1px solid #ef4444' : undefined }}
          >
            {simulationRunning ? (
              <>
                <XCircle size={15} color="#ef4444" />
                <span>Stop Simulation</span>
              </>
            ) : (
              <>
                <Play size={15} color="white" />
                <span>Start Multi-Cam Demo Simulation</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <div className="main-content">
        {/* LEFT COLUMN: Camera Registry & Live Feeds */}
        <div className="panel">
          <div className="panel-header">
            <span>Camera Registry</span>
            <span style={{ fontSize: '11px', background: '#334155', padding: '2px 6px', borderRadius: '4px' }}>
              3 Channels
            </span>
          </div>

          <div className="panel-body">
            {cameras.map((cam) => {
              const isOnline = cam.status === 'ONLINE';
              const isSelected = selectedCameraId === cam.camera_id;

              return (
                <div 
                  key={cam.camera_id} 
                  className="camera-card"
                  style={{
                    borderColor: isSelected ? '#06b6d4' : undefined,
                    cursor: 'pointer'
                  }}
                  onClick={() => setSelectedCameraId(cam.camera_id)}
                >
                  {/* Card Header */}
                  <div style={{ padding: '10px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.2)' }}>
                    <div>
                      <span style={{ fontWeight: '700', fontSize: '14px', marginRight: '6px' }}>{cam.camera_id}</span>
                      <span style={{ fontSize: '13px', color: '#cbd5e1' }}>{cam.camera_name}</span>
                    </div>
                    {isOnline ? (
                      <span className="badge-online">
                        <span className="dot-pulse"></span>
                        ONLINE
                      </span>
                    ) : (
                      <span className="badge-offline">
                        OFFLINE
                      </span>
                    )}
                  </div>

                  {/* Video Preview */}
                  <div className="camera-video-wrapper">
                    {isOnline ? (
                      <img 
                        src={`/api/cameras/${cam.camera_id}/stream`} 
                        alt={cam.camera_id}
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = 'none';
                        }}
                      />
                    ) : (
                      <div style={{ textAlign: 'center', color: '#64748b' }}>
                        <Camera size={32} style={{ margin: '0 auto 6px auto', display: 'block' }} />
                        <span style={{ fontSize: '12px' }}>Awaiting Stream Connection</span>
                      </div>
                    )}
                  </div>

                  {/* Metrics Footer */}
                  <div style={{ padding: '8px 12px', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8' }}>
                    <span>FPS: <strong style={{ color: '#f8fafc' }}>{cam.fps.toFixed(1)}</strong></span>
                    <span>Indexed: <strong style={{ color: '#f8fafc' }}>{cam.frame_count}</strong></span>
                    <span>Last: <strong style={{ color: '#f8fafc' }}>{cam.last_seen > 0 ? `${Math.round(Date.now()/1000 - cam.last_seen)}s ago` : 'never'}</strong></span>
                  </div>
                </div>
              );
            })}

            {analysisStatus?.enabled && (
              <section style={{ margin: '14px 0', padding: '12px', background: '#111c30', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <strong style={{ fontSize: '13px' }}>Analysis / Model</strong>
                  <span style={{ fontSize: '11px', color: analysisStatus.active ? '#34d399' : '#94a3b8' }}>
                    {analysisStatus.active ? 'Running' : 'Stopped'}
                  </span>
                </div>
                <label style={{ display: 'block', color: '#94a3b8', fontSize: '11px', marginBottom: '4px' }}>Input</label>
                <select
                  value={analysisInput}
                  disabled={analysisStatus.active}
                  onChange={(event) => {
                    setAnalysisInput(event.target.value);
                    if (event.target.value === 'upload') setQuerySource('uploaded');
                  }}
                  style={{ width: '100%', padding: '7px', marginBottom: '8px', background: '#1e293b', color: '#f8fafc', border: '1px solid #475569', borderRadius: '5px' }}
                >
                  <option value="upload">Upload (existing pipeline)</option>
                  {cameras.map((camera) => (
                    <option key={camera.camera_id} value={camera.camera_id}>
                      {camera.camera_id} · {camera.camera_name}
                    </option>
                  ))}
                </select>
                {analysisInput === 'upload' ? (
                  <div style={{ marginBottom: '8px', color: '#94a3b8', fontSize: '11px' }}>
                    Upload analysis continues to use the existing upload pipeline.
                    <button
                      className="btn btn-secondary"
                      onClick={() => uploadInputRef.current?.click()}
                      disabled={uploadingVideo || uploadedVideo?.status === 'processing'}
                      style={{ width: '100%', marginTop: '7px', padding: '6px 9px', fontSize: '11px' }}
                    >
                      <Upload size={13} /> Choose video
                    </button>
                  </div>
                ) : (
                  <>
                    <label style={{ display: 'block', color: '#94a3b8', fontSize: '11px', marginBottom: '4px' }}>Model</label>
                    <select
                      value={analysisModel}
                      disabled={analysisStatus.active}
                      onChange={(event) => setAnalysisModel(event.target.value as 'existing' | 'yolo11n')}
                      style={{ width: '100%', padding: '7px', marginBottom: '8px', background: '#1e293b', color: '#f8fafc', border: '1px solid #475569', borderRadius: '5px' }}
                    >
                      <option value="existing">Existing Model</option>
                      <option value="yolo11n">YOLO11n + OpenCV (ByteTrack)</option>
                    </select>
                    <button
                      className={`btn ${analysisStatus.active ? 'btn-secondary' : 'btn-primary'}`}
                      onClick={handleAnalysisToggle}
                      style={{ width: '100%', padding: '6px 9px', fontSize: '11px' }}
                    >
                      {analysisStatus.active ? 'Stop Analysis' : 'Start Live Analysis'}
                    </button>
                  </>
                )}
                {analysisActionError && <p style={{ color: '#f87171', fontSize: '11px', marginTop: '7px' }}>{analysisActionError}</p>}
                {analysisStatus.active && analysisStatus.camera_id && (
                  <div style={{ marginTop: '10px' }}>
                    <div className="camera-video-wrapper">
                      {cameras.find((camera) => camera.camera_id === analysisStatus.camera_id)?.status === 'ONLINE' ? (
                        <>
                          <img src={`/api/cameras/${analysisStatus.camera_id}/stream`} alt={`${analysisStatus.camera_id} analysis feed`} />
                          {analysisStatus.frame_width > 0 && analysisStatus.frame_height > 0 && (
                            <svg
                              viewBox={`0 0 ${analysisStatus.frame_width} ${analysisStatus.frame_height}`}
                              preserveAspectRatio="none"
                              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                            >
                              {analysisStatus.detections.map((detection, index) => {
                                const [x1, y1, x2, y2] = detection.box;
                                return (
                                  <g key={`${detection.tracking_id ?? detection.label}-${index}`}>
                                    <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke="#22d3ee" strokeWidth={2} />
                                    <text x={x1} y={Math.max(y1 - 4, 14)} fill="#22d3ee" fontSize={14}>
                                      {detection.label} {Math.round(detection.confidence * 100)}%
                                      {detection.tracking_id !== null ? ` · #${detection.tracking_id}` : ''}
                                    </text>
                                  </g>
                                );
                              })}
                            </svg>
                          )}
                        </>
                      ) : (
                        <span style={{ color: '#94a3b8', fontSize: '11px' }}>Waiting for a live frame…</span>
                      )}
                    </div>
                    <div style={{ marginTop: '7px', color: '#94a3b8', fontSize: '11px' }}>
                      {analysisStatus.detections.length
                        ? analysisStatus.detections.map((detection) => `${detection.label} ${Math.round(detection.confidence * 100)}%${detection.tracking_id !== null ? ` #${detection.tracking_id}` : ''}`).join(' · ')
                        : 'No objects detected in the latest frame'}
                      {analysisStatus.frame_timestamp && (
                        <div>Frame: {new Date(analysisStatus.frame_timestamp * 1000).toLocaleTimeString()} · {analysisStatus.inference_fps} inference FPS</div>
                      )}
                    </div>
                    {analysisStatus.error && <p style={{ color: '#f87171', fontSize: '11px', marginTop: '6px' }}>{analysisStatus.error}</p>}
                  </div>
                )}
                {analysisStatus.notifications_enabled && (
                  <div style={{ marginTop: '9px', color: '#94a3b8', fontSize: '11px' }}>
                    n8n: <strong style={{ color: analysisStatus.notification_status === 'connected' ? '#34d399' : '#fbbf24' }}>{analysisStatus.notification_status}</strong>
                    {' · '}Cooldown {analysisStatus.cooldown_seconds}s
                    {analysisStatus.last_notification && <div>Last: {analysisStatus.last_notification.event_type} · {analysisStatus.last_notification.status}</div>}
                  </div>
                )}
                <div style={{ marginTop: '9px', fontSize: '11px' }}>
                  <strong style={{ color: '#cbd5e1' }}>Recent events</strong>
                  {analysisEvents.length ? analysisEvents.slice(0, 5).map((item, index) => (
                    <div key={item.id ?? `${item.timestamp}-${index}`} style={{ marginTop: '4px', color: '#94a3b8' }}>
                      {item.event_type} · {item.objects.join(', ')} · {new Date(item.timestamp * 1000).toLocaleTimeString()}
                    </div>
                  )) : <div style={{ marginTop: '4px', color: '#64748b' }}>No events recorded.</div>}
                </div>
              </section>
            )}

            <div style={{ padding: '12px', background: 'rgba(30, 41, 59, 0.4)', borderRadius: '6px', fontSize: '11px', color: '#94a3b8', marginTop: '10px' }}>
              <strong style={{ color: '#cbd5e1' }}>LAN / Wi-Fi CCTV Clients:</strong>
              <p style={{ marginTop: '4px' }}>
                On each laptop, run its command from the project root (replace &lt;SERVER_IP&gt; with the server's LAN IP):<br/>
                <code>python server/scripts/client_cctv.py --camera-id CAM-01 --camera-name "Main Gate" --server http://&lt;SERVER_IP&gt;:8000 --device 0</code><br/>
                <code>python server/scripts/client_cctv.py --camera-id CAM-02 --camera-name "Parking Area" --server http://&lt;SERVER_IP&gt;:8000 --device 0</code><br/>
                <code>python server/scripts/client_cctv.py --camera-id CAM-03 --camera-name "Exit Gate" --server http://&lt;SERVER_IP&gt;:8000 --device 0</code><br/>
                Install client dependencies first with <code>python -m pip install requests opencv-python</code>.
              </p>
            </div>
          </div>
        </div>

        {/* CENTER COLUMN: Natural Language Query & Chat */}
        <div className="panel">
          <div className="panel-header">
            <span>Natural Language Intelligence Chat</span>
            <span style={{ fontSize: '11px', color: detectorMode === 'owlvit' ? '#10b981' : '#fbbf24', display: 'flex', alignItems: 'center', gap: '4px' }}>
              {detectorMode === 'owlvit' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
              {detectorMode === 'owlvit' ? 'Open-Vocabulary Detector Ready' : detectorMode === 'fallback' ? 'Limited Vision Fallback' : 'Vision Model Not Loaded'}
            </span>
          </div>

          <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border-color)', background: '#111c30' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', marginBottom: '8px' }}>
              <strong style={{ fontSize: '12px', color: '#cbd5e1' }}>Analyze uploaded video</strong>
              <button
                className="btn btn-secondary"
                onClick={() => uploadInputRef.current?.click()}
                disabled={uploadingVideo || uploadedVideo?.status === 'processing'}
                style={{ padding: '6px 10px', fontSize: '12px' }}
              >
                {uploadingVideo ? <RefreshCw size={13} className="animate-spin" /> : <Upload size={13} />}
                {uploadingVideo ? 'Uploading…' : 'Choose video'}
              </button>
              <input
                ref={uploadInputRef}
                type="file"
                accept="video/*,.mkv"
                onChange={handleVideoUpload}
                style={{ display: 'none' }}
              />
            </div>
            {uploadedVideo && (
              <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{uploadedVideo.filename}</span>
                  <strong style={{ color: uploadedVideo.status === 'completed' ? '#34d399' : uploadedVideo.status === 'failed' ? '#f87171' : '#fbbf24' }}>
                    {uploadedVideo.status === 'processing' ? `Processing ${uploadedVideo.progress}%` : uploadedVideo.status}
                  </strong>
                </div>
                {uploadedVideo.status === 'processing' && (
                  <div style={{ height: '3px', background: '#334155', borderRadius: '3px', marginTop: '6px' }}>
                    <div style={{ width: `${uploadedVideo.progress}%`, height: '100%', background: '#06b6d4', borderRadius: '3px' }} />
                  </div>
                )}
                {uploadedVideo.status === 'completed' && (
                  <div>
                    <span>{uploadedVideo.frame_count} frames processed · {uploadedVideo.duration_seconds.toFixed(1)} sec</span>
                    {!uploadedVideoResult && (
                      <div style={{ marginTop: '5px', color: '#cbd5e1' }}>
                        Select <strong>Uploaded video</strong>, then ask a question below to see detections and timestamps.
                      </div>
                    )}
                  </div>
                )}
                {uploadedVideo.status === 'failed' && <span style={{ color: '#f87171' }}>{uploadedVideo.error}</span>}
              </div>
            )}
            {uploadError && <p style={{ color: '#f87171', fontSize: '11px', marginBottom: '8px' }}>{uploadError}</p>}
            {querySource === 'uploaded' && uploadedVideoResult && (
              <div style={{
                marginBottom: '8px',
                padding: '10px',
                borderRadius: '6px',
                background: uploadedVideoResult.matched ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: `1px solid ${uploadedVideoResult.matched ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}`,
              }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: uploadedVideoResult.matched ? '#34d399' : '#fca5a5' }}>
                  {uploadedVideoResult.matched ? 'VIDEO ANSWER' : 'NO MATCHING EVIDENCE'}
                </div>
                <p style={{ marginTop: '5px', color: '#f8fafc', fontSize: '12px', lineHeight: 1.4 }}>
                  {uploadedVideoResult.explanation}
                </p>
                {uploadedVideoResult.matched && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginTop: '7px' }}>
                    <span style={{ color: '#cbd5e1', fontSize: '11px' }}>
                      Time {uploadedVideoResult.timestamp_iso ?? 'N/A'} · Confidence {Math.round(uploadedVideoResult.confidence * 100)}%
                    </span>
                    <button
                      className="btn btn-primary"
                      onClick={() => setSelectedEvidence(uploadedVideoResult)}
                      style={{ padding: '5px 8px', fontSize: '11px' }}
                    >
                      <Play size={12} /> View frame
                    </button>
                  </div>
                )}
              </div>
            )}
            <div style={{ display: 'flex', gap: '6px' }}>
              {([
                ['live', 'Live CCTV'],
                ['uploaded', 'Uploaded video'],
              ] as const).map(([source, label]) => (
                <button
                  key={source}
                  className={querySource === source ? 'btn btn-primary' : 'btn btn-secondary'}
                  onClick={() => {
                    setQuerySource(source);
                    setSelectedEvidence(null);
                  }}
                  disabled={source === 'uploaded' && uploadedVideo?.status !== 'completed'}
                  style={{ padding: '5px 9px', fontSize: '11px' }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Chat Feed */}
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column' }}>
            {detectorMode === 'fallback' && (
              <div style={{ marginBottom: '10px', padding: '9px 12px', borderRadius: '6px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', color: '#fbbf24', fontSize: '12px' }}>
                OWL-ViT weights are unavailable. The local fallback cannot reliably recognize clothing or person attributes, so descriptive queries may not match.
              </div>
            )}
            {messages.map((msg) => (
              <div 
                key={msg.id} 
                className={msg.sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-system'}
              >
                {/* User Message */}
                {msg.sender === 'user' && (
                  <div>{msg.text}</div>
                )}

                {/* System Message */}
                {msg.sender === 'system' && (
                  <div>
                    {msg.result ? (
                      <div>
                        {/* Status Header */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <span style={{
                            fontSize: '12px',
                            fontWeight: '700',
                            padding: '3px 8px',
                            borderRadius: '4px',
                            background: msg.result.matched ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                            color: msg.result.matched ? '#34d399' : '#f87171',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}>
                            {msg.result.matched ? (
                              <>
                                <CheckCircle2 size={13} />
                                EVIDENCE GROUNDED: YES
                              </>
                            ) : (
                              <>
                                <XCircle size={13} />
                                NO VISUAL EVIDENCE
                              </>
                            )}
                          </span>

                          {msg.result.confidence > 0 && (
                            <span style={{ fontSize: '12px', color: '#94a3b8' }}>
                              Confidence: <strong style={{ color: '#f8fafc' }}>{(msg.result.confidence * 100).toFixed(0)}%</strong>
                            </span>
                          )}
                        </div>

                        {/* Explanation text */}
                        <p style={{ margin: '8px 0', lineHeight: '1.4' }}>{msg.result.explanation}</p>

                        {/* Unresolved Location Alias Prompt */}
                        {msg.result.unresolved_alias_prompt && (
                          <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '6px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#fbbf24', fontSize: '13px', fontWeight: '600', marginBottom: '8px' }}>
                              <AlertCircle size={16} />
                              <span>Unknown Camera Location: "{msg.result.unresolved_alias_prompt}"</span>
                            </div>
                            <p style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '8px' }}>
                              Select which camera this location corresponds to (will be permanently remembered):
                            </p>
                            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                              {cameras.map(c => (
                                <button
                                  key={c.camera_id}
                                  onClick={() => handleResolveAlias(msg.result!.unresolved_alias_prompt!, c.camera_id, msg.originalQuery)}
                                  className="btn btn-secondary"
                                  style={{ fontSize: '12px', padding: '6px 10px' }}
                                >
                                  {c.camera_id} ({c.camera_name})
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Matched Evidence Summary & CTA */}
                        {msg.result.matched && (
                          <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ fontSize: '12px', color: '#94a3b8' }}>
                              <span>Camera: <strong style={{ color: '#cbd5e1' }}>{msg.result.camera_name} ({msg.result.camera_id})</strong></span>
                              <br/>
                              <span>Time: <strong style={{ color: '#cbd5e1' }}>
                                {msg.result.timestamp_iso
                                  ? (msg.result.timestamp_iso.includes('T')
                                    ? `${msg.result.timestamp_iso.split('T')[1].slice(0, 8)} UTC`
                                    : msg.result.timestamp_iso)
                                  : 'N/A'}
                              </strong></span>
                            </div>
                            <button 
                              onClick={() => setSelectedEvidence(msg.result!)}
                              className="btn btn-primary"
                              style={{ fontSize: '12px', padding: '6px 12px' }}
                            >
                              <Play size={13} />
                              <span>View Evidence</span>
                            </button>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p>{msg.text}</p>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Example query chips */}
          <div style={{ padding: '8px 16px', background: '#0f172a', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '6px', overflowX: 'auto' }}>
            <span style={{ fontSize: '11px', color: '#64748b', alignSelf: 'center', whiteSpace: 'nowrap' }}>Try:</span>
            {[
              ...(querySource === 'uploaded'
                ? [
                    'Did anyone use a mobile phone?',
                    'Was a person detected?',
                    'What vehicles appear in the video?',
                  ]
                : [
                    'Did a red car pass through the main gate in the last hour?',
                    'Was there a person with a blue shirt in the parking area?',
                    'Did a white truck exit through the exit gate?',
                    'Has anyone left a large bag at the main gate?',
                    'What happened at the north gate?',
                  ])
            ].map((sample, idx) => (
              <button
                key={idx}
                onClick={() => handleSendQuery(sample)}
                style={{
                  background: 'rgba(30, 41, 59, 0.7)',
                  border: '1px solid #334155',
                  color: '#cbd5e1',
                  borderRadius: '12px',
                  padding: '4px 10px',
                  fontSize: '11px',
                  whiteSpace: 'nowrap',
                  cursor: 'pointer'
                }}
              >
                {sample}
              </button>
            ))}
          </div>

          {/* Query Input Box */}
          <div style={{ padding: '14px 16px', background: '#0f172a', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '10px' }}>
            <input 
              type="text"
              placeholder={querySource === 'uploaded'
                ? 'Ask about the uploaded video (e.g. “Did anyone use a mobile phone?”)…'
                : "Ask a natural-language question (e.g. 'Did a red car pass through the main gate?')..."}
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSendQuery();
              }}
              style={{
                flex: 1,
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '10px 14px',
                color: 'white',
                fontSize: '14px',
                outline: 'none'
              }}
            />
            <button 
              onClick={() => handleSendQuery()}
              disabled={
                loadingQuery
                || !queryInput.trim()
                || (querySource === 'uploaded' && uploadedVideo?.status !== 'completed')
              }
              className="btn btn-primary"
              style={{ minWidth: '85px' }}
            >
              {loadingQuery ? <RefreshCw size={16} className="animate-spin" /> : <><Send size={15} /> Send</>}
            </button>
          </div>
        </div>

        {/* RIGHT COLUMN: Grounded Evidence Inspector */}
        <div className="panel">
          <div className="panel-header">
            <span>Visual Evidence Inspector</span>
            {selectedEvidence && (
              <span style={{ fontSize: '11px', color: '#06b6d4' }}>
                {selectedEvidence.camera_id}
              </span>
            )}
          </div>

          <div className="panel-body">
            {selectedEvidence ? (
              <div>
                {/* Evidence Card */}
                <div style={{ background: '#1e293b', borderRadius: '8px', border: '1px solid #334155', overflow: 'hidden', marginBottom: '16px' }}>
                  <div style={{ padding: '10px 14px', background: 'rgba(0,0,0,0.3)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: '600', fontSize: '13px' }}>Matching Frame</span>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>{selectedEvidence.frame_id}</span>
                  </div>

                  {/* Frame Image Preview with Bounding Box Overlay */}
                  <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', background: '#000' }}>
                    {selectedEvidence.frame_url ? (
                      <img 
                        key={`${selectedEvidence.source || 'cctv'}-${selectedEvidence.frame_id || selectedEvidence.frame_url}`}
                        src={selectedEvidence.frame_url} 
                        alt="Evidence Frame"
                        decoding="async"
                        fetchPriority="high"
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
                        No Image
                      </div>
                    )}

                    {/* Detections visualization */}
                    {selectedEvidence.detections.map((det, idx) => {
                      // Normalize coords assuming 640x360 frame
                      const x1 = (det.box[0] / 640) * 100;
                      const y1 = (det.box[1] / 360) * 100;
                      const w = ((det.box[2] - det.box[0]) / 640) * 100;
                      const h = ((det.box[3] - det.box[1]) / 360) * 100;

                      return (
                        <div
                          key={idx}
                          style={{
                            position: 'absolute',
                            left: `${x1}%`,
                            top: `${y1}%`,
                            width: `${w}%`,
                            height: `${h}%`,
                            border: '2px solid #ef4444',
                            background: 'rgba(239, 68, 68, 0.15)',
                            pointerEvents: 'none',
                          }}
                        >
                          <span style={{
                            position: 'absolute',
                            top: '-20px',
                            left: '0',
                            background: '#ef4444',
                            color: 'white',
                            fontSize: '10px',
                            fontWeight: '700',
                            padding: '1px 4px',
                            borderRadius: '2px',
                            whiteSpace: 'nowrap'
                          }}>
                            {det.label.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Evidence Video Clip */}
                {selectedEvidence.clip_url ? (
                  <div style={{ background: '#1e293b', borderRadius: '8px', border: '1px solid #334155', overflow: 'hidden', marginBottom: '16px' }}>
                    <div style={{ padding: '10px 14px', background: 'rgba(0,0,0,0.3)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '600', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Video size={14} color="#06b6d4" />
                        Incident Clip (±5 Seconds)
                      </span>
                      <span style={{ fontSize: '10px', background: 'rgba(6, 182, 212, 0.2)', color: '#06b6d4', padding: '2px 6px', borderRadius: '4px' }}>
                        H.264 / MP4
                      </span>
                    </div>

                    <div style={{ width: '100%', aspectRatio: '16/9', background: '#000' }}>
                      <video 
                        src={selectedEvidence.clip_url} 
                        controls 
                        autoPlay 
                        loop 
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: '12px', background: 'rgba(30, 41, 59, 0.5)', borderRadius: '6px', fontSize: '12px', color: '#94a3b8', marginBottom: '16px' }}>
                    <Video size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} />
                    Video clip generated from verified frame.
                  </div>
                )}

                {/* Grounding Verification Metrics */}
                <div style={{ background: '#1e293b', borderRadius: '8px', border: '1px solid #334155', padding: '14px' }}>
                  <h4 style={{ fontSize: '12px', textTransform: 'uppercase', color: '#94a3b8', marginBottom: '10px' }}>
                    Grounding Verification
                  </h4>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span>Detection Confidence:</span>
                        <strong>{(selectedEvidence.confidence * 100).toFixed(0)}%</strong>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${selectedEvidence.confidence * 100}%`, height: '100%', background: '#10b981' }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span>Semantic Relevance:</span>
                        <strong>{(selectedEvidence.relevance * 100).toFixed(0)}%</strong>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#334155', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${selectedEvidence.relevance * 100}%`, height: '100%', background: '#3b82f6' }} />
                      </div>
                    </div>

                    <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #334155' }}>
                      <span style={{ color: '#94a3b8' }}>Camera: </span>
                      <strong style={{ color: '#f8fafc' }}>{selectedEvidence.camera_name} ({selectedEvidence.camera_id})</strong>
                    </div>

                    <div>
                      <span style={{ color: '#94a3b8' }}>Timestamp: </span>
                      <strong style={{ color: '#f8fafc' }}>
                        {selectedEvidence.timestamp_iso?.includes('T')
                          ? selectedEvidence.timestamp_iso
                          : selectedEvidence.timestamp_iso || 'N/A'}
                      </strong>
                    </div>
                    {!selectedEvidence.is_live_evidence && selectedEvidence.source_age_seconds !== undefined && (
                      <div style={{ color: '#fbbf24', marginTop: '6px' }}>
                        Historical indexed frame · {(selectedEvidence.source_age_seconds / 60).toFixed(1)} minutes old
                      </div>
                    )}

                    {selectedTimeline.length > 0 && (
                      <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #334155' }}>
                        <div style={{ color: '#94a3b8', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '8px' }}>
                          Cross-camera continuity
                        </div>
                        {selectedTimeline.map((step, i) => (
                          <div key={`${step.camera_id}-${step.timestamp}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#cbd5e1', marginBottom: '4px' }}>
                            <span style={{ color: '#06b6d4', fontWeight: 700 }}>{step.camera_id}</span>
                            <span>{step.camera_name}</span>
                            <span style={{ color: '#94a3b8' }}>{new Date(step.timestamp_iso || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}</span>
                            {i < selectedTimeline.length - 1 && <span style={{ color: '#94a3b8' }}>→</span>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: '#64748b' }}>
                <ShieldAlert size={40} style={{ margin: '0 auto 12px auto', opacity: 0.6 }} />
                <h3 style={{ fontSize: '14px', color: '#cbd5e1', marginBottom: '6px' }}>No Evidence Inspected</h3>
                <p style={{ fontSize: '12px' }}>
                  Ask a question in the chat or select a previous query result to inspect visual proof with bounding boxes and video clips.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
