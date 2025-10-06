declare module "react-step-progress-bar" {
    import React from "react";
  
    export interface ProgressBarProps {
      percent: number;
      filledBackground?: string;
      unfilledBackground?: string;
      height?: string;
      children?: React.ReactNode;
    }
  
    export interface StepProps {
      children: (props: { accomplished: boolean; index: number }) => React.ReactNode;
    }
  
    export const ProgressBar: React.FC<ProgressBarProps>;
    export const Step: React.FC<StepProps>;
  }