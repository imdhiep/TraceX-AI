export type CandidateTrackletAction = {
  actionLabel: string;
  kineticsLabel: string | null;
  confidence: number;
};

export type CandidateTrackletEmbeddingInfo = {
  hasEmbedding: boolean;
  dim: number | null;
  model: string;
};

export type CandidateTracklet = {
  trackletId: string;
  videoId: string | null;
  cameraId: string | null;
  trackId: string | null;

  timeStart: string | null;
  timeEnd: string | null;
  startOffsetSeconds: number | null;
  endOffsetSeconds: number | null;
  durationSeconds: number | null;

  cropUrl: string | null;
  representativeBbox: number[] | null;

  qualityScore: number | null;
  confidence: number | null;

  gender: string | null;
  genderConf: number | null;
  ageRange: string | null;
  ageRangeConf: number | null;

  upperColor: string | null;
  upperType: string | null;
  upperDesc: string | null;
  upperConf: number | null;

  lowerColor: string | null;
  lowerType: string | null;
  lowerDesc: string | null;
  lowerConf: number | null;

  shoesColor: string | null;
  shoesType: string | null;
  shoesDesc: string | null;
  shoesConf: number | null;

  bagPresence: string | null;
  bagType: string | null;
  bagDesc: string | null;
  bagConf: number | null;

  hatPresence: string | null;
  hatColor: string | null;
  hatType: string | null;
  hatDesc: string | null;
  hatConf: number | null;

  maskPresence: string | null;
  maskConf: number | null;

  hairStyle: string | null;
  hairStyleConf: number | null;
  hairColor: string | null;
  hairColorConf: number | null;

  appearanceSummary: string | null;
  appearanceSummaryConf: number | null;

  bevX: number | null;
  bevY: number | null;

  actions: CandidateTrackletAction[];
  embedding: CandidateTrackletEmbeddingInfo;
};

export type CandidateDetail = {
  candidateId: string;
  candidateKey: string | null;
  fusionScore: number | null;
  vectorScore: number | null;
  textScore: number | null;
  appearanceSummary: string | null;
  previewUrl: string | null;
  rankPosition: number | null;
  totalTrackletsInWindow: number;
  tracklets: CandidateTracklet[];
  cameraPath: string[];
};

export type TraceSegment = {
  segmentOrder: number;
  trackletId: string;
  cameraId: string | null;
  timeStart: string | null;
  timeEnd: string | null;
  durationSeconds: number | null;
  thumbnailUrl: string | null;
  videoClipUrl: string | null;
  confidence: number | null;
};

export type BuildTraceResult = {
  evidenceId: number;
  queryId: string;
  candidateId: string;
  traceConfidence: number | null;
  segmentCount: number;
  totalDurationSeconds: number | null;
  segments: TraceSegment[];
  mergedVideoUrl: string | null;
  timeWindowStart: string | null;
  timeWindowEnd: string | null;
};
