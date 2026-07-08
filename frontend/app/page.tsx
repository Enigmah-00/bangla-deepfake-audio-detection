"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  Download,
  FileAudio,
  Mic,
  RefreshCw,
  ShieldCheck,
  Square,
  Upload,
} from "lucide-react";
import { ChangeEvent, useMemo, useRef, useState } from "react";

type EvidenceGroup = {
  key: string;
  label: string;
  value: number;
  percent: number;
};

type AnalysisResponse = {
  ok: boolean;
  audio: {
    filename: string;
    duration_seconds: number;
    analyzed_seconds: number;
    sample_rate: number;
    rms: number;
    peak: number;
  };
  prediction: {
    label: string;
    label_bn: string;
    fake_probability: number;
    real_probability: number;
    band: string;
    summary: string;
    model_score_name: string;
    disclaimer: string;
  };
  evidence: {
    title: string;
    method: string;
    groups: EvidenceGroup[];
    caution: string;
  };
  visualizations: {
    waveform: string;
    mel_spectrogram: string;
    evidence: string;
    probabilities: string;
  };
  report: Record<string, unknown>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function resultTone(band?: string) {
  if (band === "high-risk") return "tone-risk";
  if (band === "low-risk") return "tone-clear";
  return "tone-uncertain";
}

function extractErrorMessage(payload: unknown): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      const message = (detail as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
  }
  return "The analysis could not be completed. Please try another supported audio file.";
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordingName, setRecordingName] = useState<string>("");
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const selectedAudio = useMemo(() => {
    if (recordedBlob) {
      return {
        blob: recordedBlob,
        name: recordingName || "microphone_recording.webm",
        label: "Microphone recording",
      };
    }
    if (file) {
      return {
        blob: file,
        name: file.name,
        label: file.name,
      };
    }
    return null;
  }, [file, recordedBlob, recordingName]);

  function resetOutput() {
    setResult(null);
    setError("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setRecordedBlob(null);
    setRecordingName("");
    resetOutput();
  }

  async function startRecording() {
    resetOutput();
    setFile(null);
    setRecordedBlob(null);
    setRecordingName("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const preferredType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/ogg")
          ? "audio/ogg"
          : "";
      const recorder = new MediaRecorder(stream, preferredType ? { mimeType: preferredType } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const extension = mimeType.includes("ogg") ? "ogg" : "webm";
        setRecordedBlob(blob);
        setRecordingName(`microphone_recording.${extension}`);
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setError("Microphone access was not available. You can still upload an audio file.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  }

  async function analyze() {
    if (!selectedAudio) return;
    setIsAnalyzing(true);
    setError("");
    setResult(null);
    const formData = new FormData();
    formData.append("file", selectedAudio.blob, selectedAudio.name);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload));
      }
      setResult(payload as AnalysisResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The analysis could not be completed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function downloadReport() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `konthoprohori-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Track E · National Defence</p>
          <h1>KonthoProhori</h1>
          <p className="subtitle">Bangla Deepfake Audio Detection and Voice Forensics</p>
          <p className="subtitle bn">বাংলা কণ্ঠস্বর ডিপফেক শনাক্তকরণ ও ফরেনসিক বিশ্লেষণ</p>
        </div>
        <div className="status-pill">
          <ShieldCheck size={18} aria-hidden />
          AI-assisted screening
        </div>
      </header>

      <section className="workspace" aria-label="Audio analyzer">
        <div className="tool-panel">
          <div className="panel-heading">
            <FileAudio size={22} aria-hidden />
            <div>
              <h2>Analyze Audio</h2>
              <p>Upload a clip or record directly from the browser.</p>
            </div>
          </div>

          <label className="upload-zone">
            <Upload size={26} aria-hidden />
            <span>Choose audio file</span>
            <small>WAV, MP3, FLAC, OGG, M4A, WEBM · max 20 MB</small>
            <input
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
              onChange={handleFileChange}
              disabled={isAnalyzing || isRecording}
            />
          </label>

          <div className="record-row">
            <button
              className="secondary-button"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isAnalyzing}
              type="button"
            >
              {isRecording ? <Square size={18} aria-hidden /> : <Mic size={18} aria-hidden />}
              {isRecording ? "Stop recording" : "Record microphone"}
            </button>
            <span className="source-label">{selectedAudio?.label || "No audio selected"}</span>
          </div>

          {error ? (
            <div className="error-box" role="alert">
              <AlertTriangle size={18} aria-hidden />
              {error}
            </div>
          ) : null}

          <button
            className="primary-button"
            onClick={analyze}
            disabled={!selectedAudio || isAnalyzing || isRecording}
            type="button"
          >
            {isAnalyzing ? <RefreshCw className="spin" size={19} aria-hidden /> : <Activity size={19} aria-hidden />}
            {isAnalyzing ? "Analyzing..." : "Analyze clip"}
          </button>

          <div className="privacy-note">
            Audio is processed for this request only. The report excludes raw audio and server file paths.
          </div>
        </div>

        <div className={`result-panel ${resultTone(result?.prediction.band)}`}>
          {result ? (
            <>
              <div className="result-heading">
                <div>
                  <p className="eyebrow">Screening result</p>
                  <h2>{result.prediction.label}</h2>
                  <p className="bn strong">{result.prediction.label_bn}</p>
                </div>
                <button className="icon-button" onClick={downloadReport} type="button" aria-label="Download JSON report">
                  <Download size={19} aria-hidden />
                </button>
              </div>
              <p className="result-summary">{result.prediction.summary}</p>

              <div className="probability-grid">
                <div className="metric">
                  <span>Genuine model score</span>
                  <strong>{percentage(result.prediction.real_probability)}</strong>
                  <div className="meter">
                    <i style={{ width: percentage(result.prediction.real_probability) }} />
                  </div>
                </div>
                <div className="metric">
                  <span>Deepfake model score</span>
                  <strong>{percentage(result.prediction.fake_probability)}</strong>
                  <div className="meter amber">
                    <i style={{ width: percentage(result.prediction.fake_probability) }} />
                  </div>
                </div>
              </div>

              <dl className="audio-facts">
                <div>
                  <dt>Filename</dt>
                  <dd>{result.audio.filename}</dd>
                </div>
                <div>
                  <dt>Duration</dt>
                  <dd>{result.audio.duration_seconds.toFixed(2)}s</dd>
                </div>
                <div>
                  <dt>Analyzed</dt>
                  <dd>{result.audio.analyzed_seconds.toFixed(2)}s</dd>
                </div>
                <div>
                  <dt>Sample rate</dt>
                  <dd>{result.audio.sample_rate} Hz</dd>
                </div>
              </dl>
            </>
          ) : (
            <div className="empty-state">
              <BarChart3 size={32} aria-hidden />
              <h2>Awaiting audio</h2>
              <p>The model output, probabilities, waveform, spectrogram, and acoustic evidence will appear here.</p>
            </div>
          )}
        </div>
      </section>

      {result ? (
        <section className="charts" aria-label="Analysis visualizations">
          <figure className="chart-tile">
            <img src={result.visualizations.waveform} alt="Waveform of analyzed audio" />
          </figure>
          <figure className="chart-tile">
            <img src={result.visualizations.mel_spectrogram} alt="Log-mel spectrogram of analyzed audio" />
          </figure>
          <figure className="chart-tile">
            <img src={result.visualizations.probabilities} alt="Genuine and deepfake model probabilities" />
          </figure>
          <figure className="chart-tile">
            <img src={result.visualizations.evidence} alt="Grouped acoustic evidence chart" />
          </figure>
          <div className="evidence-list">
            <h2>Acoustic Evidence</h2>
            <p>{result.evidence.caution}</p>
            {result.evidence.groups.map((group) => (
              <div className="evidence-row" key={group.key}>
                <span>{group.label}</span>
                <strong>{group.percent.toFixed(1)}%</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="info-grid">
        <article>
          <h2>How It Works</h2>
          <p>
            The backend converts audio to mono 16 kHz speech, analyzes up to the first five seconds, extracts the
            notebook-compatible 227 acoustic features, applies the saved StandardScaler, then scores the trained XGBoost
            classifier.
          </p>
        </article>
        <article>
          <h2>Intended Use</h2>
          <p>
            KonthoProhori supports first-line screening for journalists, fact-checkers, security teams, moderators, and
            citizens evaluating suspicious Bangla voice clips.
          </p>
        </article>
        <article>
          <h2>Limitations</h2>
          <p>
            This is not legal or forensic proof. Performance may shift for unseen speakers, generators, accents,
            compression, background noise, and recording devices.
          </p>
        </article>
        <article>
          <h2>Attribution</h2>
          <p>
            Dataset reference: Bangla DeepFake Dataset by ahanaf101 on Kaggle. The reported 99.57% result is from the
            notebook random held-out split, not a universal field-accuracy claim.
          </p>
        </article>
      </section>
    </main>
  );
}

