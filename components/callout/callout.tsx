import type {ReactElement, ReactNode} from "react";
import cn from "clsx";
import {VscRocket, VscInfo, VscWarning, VscError, VscBeaker, VscSymbolEvent} from "react-icons/vsc";


const TypeToEmoji = {
    default: <VscRocket className="w-[28px] h-[28px]"/>,
    info: <VscInfo className="w-[28px] h-[28px]"/>,
    warning: <VscWarning className="w-[28px] h-[28px]"/>,
    error: <VscError className="w-[28px] h-[28px]"/>,
    example: <VscBeaker className="w-[28px] h-[28px]"/>,
    todo: <VscSymbolEvent className="w-[28px] h-[28px]"/>
};

type CalloutType = keyof typeof TypeToEmoji

const classes: Record<CalloutType, string> = {
    default: cn(
        "border-primary-200 bg-primary-50 text-primary-800 dark:border-primary-400/30 dark:bg-primary-400/20 dark:text-primary-300"
    ),
    error: cn(
        "border-red-300 bg-red-100 text-red-900 dark:border-red-200/30 dark:bg-red-900/30 dark:text-red-200"
    ),
    info: cn(
        "border-blue-300 bg-blue-100 text-blue-900 dark:border-blue-200/30 dark:bg-blue-900/30 dark:text-blue-200"
    ),
    warning: cn(
        "border-yellow-200 bg-yellow-50 text-yellow-900 dark:border-yellow-200/30 dark:bg-yellow-700/30 dark:text-yellow-200"
    ),
    example: cn(
        "border-primary-200 bg-primary-50 text-primary-800 dark:border-primary-400/30 dark:bg-primary-400/20 dark:text-primary-300"
    ),
    todo: cn(
        "border-yellow-100 bg-yellow-50 text-yellow-900 dark:border-yellow-200/30 dark:bg-yellow-700/30 dark:text-yellow-200"
    )
};

type CalloutProps = {
    type?: CalloutType
    emoji?: string | ReactElement
    children: ReactNode
}

export default function Callout({
    children,
    type = "default",
    emoji = TypeToEmoji[type]
}: CalloutProps): ReactElement {
    return (
        <div
            className={cn(
                "nextra-callout mt-6 flex rounded-lg border py-2 ltr:pr-4 rtl:pl-4",
                "contrast-more:border-current contrast-more:dark:border-current",
                classes[type]
            )}
        >
            <div
                className="select-none ltr:pr-2 rtl:pr-3 rtl:pl-2 ltr:pl-3 text-xl font-bold capitalize"
                style={{
                    fontFamily: "\"Apple Color Emoji\", \"Segoe UI Emoji\", \"Segoe UI Symbol\""
                }}
            >
                {emoji}
            </div>
            <div className="nx-w-full nx-min-w-0 nx-leading-7">{children}</div>
        </div>
    );
}
