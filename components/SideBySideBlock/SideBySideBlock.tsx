export const SideBySideBlock = ({ children } : { children: React.ReactNode }) => {
  return (
    <div className="flex flex-row flex-wrap justify-between w-full mt-6 -mb-4">
      {children}
    </div>
  );
}

export const Block = ({ children } : { children: React.ReactNode }) => {
    return (
        <div className="w-full xl:w-[calc(50%-0.5rem)] mb-0">
            {children}
        </div>
    );
}