export interface EvalRow {
  /** Spec milestone, e.g. "M2". */
  milestone: string;
  /** What is checked, as the spec words it. */
  test: string;
  /** Pass bar, printed as-is. */
  target: string;
  /** What this run measured, printed as-is. */
  measured: string;
  /** null means the check could not run here (and `note` says why). */
  pass: boolean | null;
  note?: string;
}

export type Section = (opts: { quick: boolean; log: (line: string) => void }) => Promise<EvalRow[]>;
