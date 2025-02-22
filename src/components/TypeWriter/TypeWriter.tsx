"use client";

import { ReactTyped } from "react-typed";
import { ComponentProps } from "react";

type TypeWriterProps = ComponentProps<typeof ReactTyped>;

export default function TypeWriter(props: TypeWriterProps) {
  return <ReactTyped {...props} />;
}
