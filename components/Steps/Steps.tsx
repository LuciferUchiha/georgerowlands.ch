import React, { ReactNode } from 'react';

interface StepCircleProps {
  id: string;
  scale?: number; // Optional scale prop (default is 1)
}

const StepCircle: React.FC<StepCircleProps> = ({ id, scale = 1 }) => {
  const circleStyle = {
    transform: `scale(${scale})`,
  };

  return (
    <span className="inline-flex h-8 w-8 rounded-full bg-gray-100 dark:bg-neutral-800 items-center justify-center text-neutral-400 border border-gray-200 dark:border-neutral-800" style={circleStyle}>
      {id}
    </span>
  );
};

interface StepProps {
  children: ReactNode;
  id: string;
}

const Step: React.FC<StepProps> = ({ children, id }) => {
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

interface StepsProps {
  children: ReactNode;
}

const Steps: React.FC<StepsProps> = ({ children }) => {
  return (
    <div className="mt-8 space-y-4 relative">
      {children}
    </div>
  );
}

export { Steps, Step, StepCircle };
