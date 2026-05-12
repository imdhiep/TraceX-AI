"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { saveSession } from "@/lib/auth";
import { getApiBaseUrl } from "@/lib/api";
import { mapBackendErrorMessage, parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";
import { useToast } from "@/components/ui/ToastProvider";

type AuthResponse = {
  access_token: string;
  token_type: string;
  user: {
    id: number;
    email: string;
    full_name?: string;
    role: "SUPER_ADMIN" | "ADMIN" | "USER";
    is_active: boolean;
  };
};

function friendlyErrorByStatus(status: number): string {
  if (status === 400) return "Thông tin đăng nhập không hợp lệ.";
  if (status === 401) return "Email hoặc mật khẩu không đúng.";
  if (status === 403) return "Tài khoản không có quyền truy cập.";
  if (status === 404) return "Không tìm thấy dịch vụ đăng nhập. Vui lòng thử lại sau.";
  if (status >= 500) return "Hệ thống đang bận. Vui lòng thử lại sau ít phút.";
  return "Đăng nhập thất bại. Vui lòng thử lại.";
}

async function readError(response: Response): Promise<string> {
  try {
    return await readApiErrorMessage(response);
  } catch {
    return friendlyErrorByStatus(response.status);
  }
}

async function loginWithFallback(identifier: string, password: string): Promise<Response> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/auth/login`;
  return fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: identifier, identifier, password }),
  });
}

export function LoginPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFieldErrors({});

    const identifier = email.trim();
    const nextFieldErrors: { email?: string; password?: string } = {};
    if (!identifier) {
      nextFieldErrors.email = "Email không được để trống.";
    }
    if (!password.trim()) {
      nextFieldErrors.password = "Password không được để trống.";
    }
    if (Object.keys(nextFieldErrors).length > 0) {
      setFieldErrors(nextFieldErrors);
      showToast("Vui lòng kiểm tra lại thông tin đăng nhập.", "error");
      return;
    }

    setIsLoading(true);
    try {
      const response = await loginWithFallback(identifier, password);
      if (!response.ok) {
        const errorMessage = await readError(response);
        setFieldErrors({ email: "Thông tin đăng nhập không hợp lệ.", password: "Thông tin đăng nhập không hợp lệ." });
        throw new Error(errorMessage);
      }
      const payload = await parseJsonOrThrow<AuthResponse>(response);
      saveSession(payload.access_token, payload.user);
      showToast(`Đăng nhập thành công. Xin chào ${payload.user.email}!`, "success");
      router.push("/home");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Đăng nhập thất bại.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div
      className="relative min-h-screen overflow-hidden"
      style={{
        backgroundImage: "url('/images/auth/img_backgroud5.png')",
        backgroundSize: "cover",
        backgroundPosition: "center top",
        backgroundRepeat: "no-repeat",
      }}
    >
      <div className="absolute inset-0 bg-slate-950/18" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_22%_30%,rgba(255,255,255,0.34),transparent_48%),radial-gradient(circle_at_82%_20%,rgba(174,213,255,0.26),transparent_45%)]" />


      <div className="relative mx-auto grid min-h-screen w-full max-w-7xl items-center gap-8 px-4 py-8 md:grid-cols-[1.22fr_0.78fr] md:px-10">
        <section className="self-start rounded-3xl border border-white/28 bg-gradient-to-br from-white/22 via-white/12 to-sky-200/12 p-6 pt-10 text-white shadow-[0_18px_50px_rgba(2,8,23,0.35)] backdrop-blur-xl transition-[background-color,border-color,box-shadow] duration-300 hover:border-white/45 hover:shadow-[0_24px_70px_rgba(2,8,23,0.45)] md:p-10 md:pt-20">
          <div className="flex items-center gap-3">
            <img src="/images/auth/img_icon_se.png" alt="logo" className="h-11 w-11 rounded-xl object-contain" />
            <div>
              <p className="text-3xl font-bold leading-tight text-[#F6FAFF] drop-shadow-[0_2px_6px_rgba(10,30,60,0.42)]">
                TraceX-AI
              </p>
              <p className="text-sm text-[#E8F2FF]/90">AI-Powered CCTV Search Engine Platform</p>
            </div>
          </div>

          <h1 className="mt-7 max-w-3xl text-4xl font-bold leading-tight md:text-6xl">
            <span className="text-[#F6FAFF] drop-shadow-[0_2px_6px_rgba(10,30,60,0.42)]">Hệ Thống Tìm Người,</span>
            <br />
            <span className="text-[#5DD4FF] drop-shadow-[0_2px_10px_rgba(34,200,255,0.45)]">Nhận Diện Hành Động.</span>
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-[#E8F2FF] md:text-lg">
            Nền tảng tìm kiếm và theo dõi đối tượng trên hệ thống nhiều camera. Ứng dụng trí tuệ nhân tạo để phân tích ngoại hình, nhận diện hành động và khoanh vùng vị trí một cách tự động.
          </p>

          <div className="mt-8 max-w-xl space-y-3">
            <div className="flex items-start gap-3 rounded-2xl border border-white/18 bg-white/8 p-3.5 transition-[border-color,background-color,box-shadow] duration-300 hover:border-white/35 hover:bg-white/14 hover:shadow-[0_12px_30px_rgba(2,8,23,0.28)]">
              <img src="/images/auth/img_icon_Ai.png" alt="AI" className="mt-0.5 h-12 w-12 rounded-lg object-contain" />
              <div>
                <p className="text-xl font-semibold text-[#F5FAFF]">Tìm Kiếm Đa Chiều</p>
                <p className="text-sm text-[#DFECFF]">Tìm nhanh mục tiêu bằng cách kết hợp đặc điểm ngoại hình, trang phục và hành động.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 rounded-2xl border border-white/18 bg-white/8 p-3.5 transition-[border-color,background-color,box-shadow] duration-300 hover:border-white/35 hover:bg-white/14 hover:shadow-[0_12px_30px_rgba(2,8,23,0.28)]">
              <img src="/images/auth/img_icon_cam.png" alt="Camera" className="mt-0.5 h-12 w-12 rounded-lg object-contain" />
              <div>
                <p className="text-xl font-semibold text-[#F5FAFF]">Vẽ Lại Lộ Trình</p>
                <p className="text-sm text-[#DFECFF]">Hệ thống tự động liên kết hình ảnh thu được từ các camera khác nhau để dựng lại chính xác hành trình di chuyển của đối tượng.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 rounded-2xl border border-white/18 bg-white/8 p-3.5 transition-[border-color,background-color,box-shadow] duration-300 hover:border-white/35 hover:bg-white/14 hover:shadow-[0_12px_30px_rgba(2,8,23,0.28)]">
              <img src="/images/auth/img_icon_se2.png" alt="Search" className="mt-0.5 h-12 w-12 rounded-lg object-contain" />
              <div>
                <p className="text-xl font-semibold text-[#F5FAFF]">Tốc Độ Vượt Trội</p>
                <p className="text-sm text-[#DFECFF]">Trải nghiệm tra cứu mượt mà nhờ hệ thống được tối ưu đặc biệt, trả về kết quả tìm kiếm ngay nhanh chóng.</p>
              </div>
            </div>
          </div>

          <div className="mt-7 flex flex-wrap items-center gap-6 text-sm text-[#EAF3FF]">
            <div className="flex items-center gap-2">
              <img src="/images/auth/img_icon_se2.png" alt="safe" className="h-5 w-5 rounded-md border border-white/70 bg-white/90 p-0.5 shadow-sm object-contain" />
              <span>Kết quả chuẩn xác</span>
            </div>
            <div className="flex items-center gap-2">
              <img src="/images/auth/img_icon_cloud.png" alt="cloud" className="h-5 w-5 rounded-md border border-white/70 bg-white/90 p-0.5 shadow-sm object-contain" />
              <span>Phân tích thông minh</span>
            </div>
            <div className="flex items-center gap-2">
              <img src="/images/auth/img_icon_thunder.png" alt="realtime" className="h-5 w-5 rounded-md border border-white/70 bg-white/90 p-0.5 shadow-sm object-contain" />
              <span>Phản hồi dưới 5 giây</span>
            </div>
          </div>
        </section>

        <section className="flex items-center justify-center md:justify-end">
          <div className="w-full max-w-md rounded-3xl border border-white/52 bg-gradient-to-br from-white/84 via-white/72 to-sky-100/62 p-8 shadow-[0_18px_50px_rgba(15,23,42,0.28)] backdrop-blur-2xl transition-[background-color,border-color,box-shadow] duration-300 hover:border-white/70 hover:shadow-[0_28px_70px_rgba(15,23,42,0.36)] md:p-9">
            <div className="mb-8 text-center">
              <h2 className="text-2xl font-semibold text-ink">Log in</h2>
              <p className="mt-2 text-sm text-ink-secondary">Sử dụng email và password để tiếp tục</p>
            </div>

            <form className="flex flex-col gap-4" onSubmit={handleLogin}>
              <label className="flex flex-col gap-1.5 text-sm font-medium text-ink">
                Email
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    if (fieldErrors.email) {
                      setFieldErrors((prev) => ({ ...prev, email: undefined }));
                    }
                  }}
                  autoComplete="username"
                  className={[
                    "rounded-2xl border px-4 py-3 text-base text-ink shadow-card outline-none ring-accent/25 focus:ring-2",
                    fieldErrors.email
                      ? "border-red-300 bg-red-50 focus:border-red-400"
                      : "border-surface-muted focus:border-accent",
                  ].join(" ")}
                  placeholder="name@company.com"
                  aria-invalid={Boolean(fieldErrors.email)}
                />
                {fieldErrors.email ? <span className="text-xs text-red-600">{fieldErrors.email}</span> : null}
              </label>
              <label className="flex flex-col gap-1.5 text-sm font-medium text-ink">
                Password
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      if (fieldErrors.password) {
                        setFieldErrors((prev) => ({ ...prev, password: undefined }));
                      }
                    }}
                    autoComplete="current-password"
                    className={[
                      "w-full rounded-2xl border px-4 py-3 pr-12 text-base text-ink shadow-card outline-none ring-accent/25 focus:ring-2",
                      fieldErrors.password
                        ? "border-red-300 bg-red-50 focus:border-red-400"
                        : "border-surface-muted focus:border-accent",
                    ].join(" ")}
                    placeholder="••••••••"
                    aria-invalid={Boolean(fieldErrors.password)}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((current) => !current)}
                    className="absolute right-3 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full text-ink-secondary transition hover:bg-slate-100 hover:text-ink focus:outline-none focus:ring-2 focus:ring-accent/30"
                    aria-label={showPassword ? "Ẩn mật khẩu" : "Hiển thị mật khẩu"}
                    aria-pressed={showPassword}
                  >
                    {showPassword ? (
                      <svg
                        aria-hidden="true"
                        viewBox="0 0 24 24"
                        className="h-5 w-5"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="m3 3 18 18" />
                        <path d="M10.58 10.58a2 2 0 0 0 2.83 2.83" />
                        <path d="M9.88 4.24A10.67 10.67 0 0 1 12 4c5 0 8.5 4.5 9.5 6.5a11.8 11.8 0 0 1-2.31 3.19" />
                        <path d="M6.61 6.61A13.43 13.43 0 0 0 2.5 10.5C3.5 12.5 7 17 12 17a10.83 10.83 0 0 0 4.18-.82" />
                      </svg>
                    ) : (
                      <svg
                        aria-hidden="true"
                        viewBox="0 0 24 24"
                        className="h-5 w-5"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    )}
                  </button>
                </div>
                {fieldErrors.password ? <span className="text-xs text-red-600">{fieldErrors.password}</span> : null}
              </label>

              <button
                type="submit"
                disabled={isLoading}
                className="mt-2 rounded-2xl bg-blue-600 py-3 text-sm font-semibold text-white shadow-card transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "Đang xử lý..." : "Log in"}
              </button>

              <button
                type="button"
                onClick={() => showToast("Tính năng forgot password sẽ được bổ sung sau.", "info")}
                className="rounded-2xl border border-surface-muted bg-white py-3 text-sm font-semibold text-ink-secondary transition hover:bg-surface hover:text-ink"
              >
                Forgot password
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}
