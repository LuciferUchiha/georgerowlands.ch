"use client";

import { useTheme } from "nextra-theme-docs";
import dynamic from "next/dynamic";
import { useWindowDimensions } from "~/src/utils/windowDimensions";
import Data from "~/data/graph.json";
import { useRouter } from "next/navigation";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
});

export default function GardenGraph() {
  const router = useRouter();
  const { theme, systemTheme } = useTheme();
  const currentTheme = theme !== "system" ? theme : systemTheme;
  const background = currentTheme === "dark" ? "#111" : "#fff";

  const windowDimensions = useWindowDimensions();

  if (windowDimensions.width < 1280) {
    return null;
  }

  return (
    <ForceGraph3D
      graphData={Data}
      width={750}
      height={750}
      backgroundColor={background}
      nodeRelSize={10}
      nodeColor={() => "#8000FF"}
      nodeLabel={(node) =>
        `<span>${node.name}</span>`
      }
      linkColor={() => "#8000FF"}
      linkOpacity={0.5}
      onNodeHover={(node) => {
        document.body.style.cursor = node ? "pointer" : null;
      }}
      onNodeClick={(node) => {
        if (typeof node.id === "string") {
          router.push(node.id);
        }
      }}
    />
  );
};
