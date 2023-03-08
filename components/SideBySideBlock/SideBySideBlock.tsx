import type {ReactElement, ReactNode} from "react";

type SideBySideBlockProps = {
    children: ReactNode
}

export function SideBySideBlock({
                                    children,
                                }: SideBySideBlockProps): ReactElement {
    return (
        <div className="flex flex-row flex-wrap justify-between w-full mt-6 -mb-4">
            {children}
        </div>
    );
}

type BlockProps = {
    children: ReactNode
}

export function Block({
                          children,
                      }: BlockProps): ReactElement {
    return (
        <div className="w-full xl:w-[calc(50%-0.5rem)] mb-0">
            {children}
        </div>
    );
}