const YOUTUBE_VIDEO_ID = "oU4uv5XhSNY";

export default function GuidePage() {
  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 py-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Hướng dẫn sử dụng TraceX-AI</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Xem video bên dưới để nắm các luồng chính: tìm kiếm bằng mô tả/ảnh, lọc theo camera & thời gian, chọn candidate, dựng trace evidence và xem lại lịch sử.
        </p>
      </header>

      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-black shadow-[0_18px_40px_rgba(15,23,42,0.18)] dark:border-slate-800 dark:shadow-[0_24px_50px_rgba(2,6,23,0.55)]">
        <div className="relative w-full" style={{ paddingTop: "56.25%" }}>
          <iframe
            className="absolute inset-0 h-full w-full"
            src={`https://www.youtube.com/embed/${YOUTUBE_VIDEO_ID}?rel=0&modestbranding=1`}
            title="Hướng dẫn sử dụng TraceX-AI"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            referrerPolicy="strict-origin-when-cross-origin"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <section className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-950">
          <h2 className="text-base font-semibold text-slate-900 dark:text-white">Trong video bạn sẽ thấy</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-700 dark:text-slate-300">
            <li>Tìm kiếm bằng mô tả văn bản hoặc ảnh mẫu.</li>
            <li>Lọc theo vị trí camera và khoảng thời gian.</li>
            <li>Xem candidate, preview, toàn bộ tracklet.</li>
            <li>Chọn candidate để dựng evidence/timeline.</li>
            <li>Xem lại lịch sử query và candidate đã trace.</li>
          </ul>
        </section>
        <section className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-950">
          <h2 className="text-base font-semibold text-slate-900 dark:text-white">Mẹo nhanh</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-700 dark:text-slate-300">
            <li>Kết hợp ảnh + mô tả để tăng độ tin cậy của ranking.</li>
            <li>Có thể loại tracklet sai khỏi candidate trước khi build trace.</li>
            <li>
              Xem chi tiết kỹ thuật ở mục <span className="font-medium text-slate-900 dark:text-white">Hướng dẫn</span> trong tài liệu repo.
            </li>
          </ul>
        </section>
      </div>

      <p className="text-xs text-slate-500 dark:text-slate-400">
        Nếu không xem được, mở trực tiếp tại{" "}
        <a
          href={`https://youtu.be/${YOUTUBE_VIDEO_ID}`}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-blue-600 hover:underline dark:text-blue-400"
        >
          youtu.be/{YOUTUBE_VIDEO_ID}
        </a>
        .
      </p>
    </div>
  );
}
