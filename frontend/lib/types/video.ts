export type VideoItem = {
  id: string;
  title: string;
  description: string;
  thumbnailUrl: string;
  queryId?: string;
};

export type VideoClip = VideoItem & {
  previewUrl?: string;
  durationLabel?: string;
};
