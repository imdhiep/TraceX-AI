import type { VideoItem } from "@/lib/types";

import { VideoCard } from "./VideoCard";

type VideoGridProps = {
  items: VideoItem[];
  onItemClick?: (video: VideoItem) => void;
};

export function VideoGrid({ items, onItemClick }: VideoGridProps) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
      {items.map((video) => (
        <VideoCard key={video.id} video={video} onClick={onItemClick} />
      ))}
    </div>
  );
}
