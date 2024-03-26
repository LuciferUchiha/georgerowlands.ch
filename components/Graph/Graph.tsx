import { useTheme } from "nextra-theme-docs";
import Data from "./network.json";
import dynamic from "next/dynamic";
import { useEffect } from "react";
import { useWindowDimensions } from "~utils/windowDimensions";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
});

const Graph = () => {
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
      width={800}
      height={800}
      backgroundColor={background}
    />
  );
};

export default Graph;
