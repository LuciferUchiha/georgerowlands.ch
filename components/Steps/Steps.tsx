import React, { ReactNode } from 'react';

const StepCircle = ({ id, scale = 1 } : { id: string, scale?: number }) => {
  const circleStyle = {
    transform: `scale(${scale})`,
  };

  return (
    <span className="inline-flex h-8 w-8 rounded-full bg-gray-100 dark:bg-neutral-800 items-center justify-center text-neutral-400 border border-gray-200 dark:border-neutral-800" style={circleStyle}>
      {id}
    </span>
  );
};

export const Step = ({ id, children }: { id: string, children: ReactNode }) => {
  return (
    <div className="flex mb-4 items-start relative">
      <div className="flex-shrink-0 mt-1 mr-2">
        <StepCircle id={id} />
      </div>
      <div className="mt-1.5">
        {children}
      </div>
    </div>
  );
};

export const Steps = ({ children }: { children: ReactNode }) => {
  return (
    <div className="mt-8 space-y-4 relative">
      {children}
    </div>
  );
}