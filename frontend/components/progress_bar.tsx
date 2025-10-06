import React from "react";

export default function TheProgressBar({ step }: { step: string }) {
  const steps_map:string[] = ["idle", "file-selected", "processing", "completed"] as const;
  const currentIndex = steps_map.indexOf(step);
  const current = currentIndex >= 0 ? currentIndex + 1 : 1;
  const steps = [1, 2, 3, 4];

  // calculate progress width (0% → 100%)
  const progressWidth = ((current - 1) / (steps.length - 1)) * 100 + "%";

  return (
    <div className="the-progressbar">
      {/* Background bar */}
      <div className="tpb-bg" />
      {/* Filled portion */}
      <div className="tpb-fill" style={{ width: progressWidth }} />
      {/* Circles */}
      {steps.map((s) => (
        <div key={s} className={`tpb-circle ${s <= current ? "achieved" : ""}`}>
          {s}
        </div>
      ))}
    </div>
  );
}
