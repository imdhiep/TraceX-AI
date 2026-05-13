import { TraceView } from "@/features/trace/TraceView";

type PageProps = {
  params: { evidenceId: string };
  searchParams?: { query?: string; candidate?: string };
};

export default function TracePage({ params, searchParams }: PageProps) {
  return (
    <TraceView
      evidenceId={Number(params.evidenceId)}
      queryId={searchParams?.query ?? null}
      candidateId={searchParams?.candidate ?? null}
    />
  );
}
