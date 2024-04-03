import { useTheme } from "nextra-theme-docs";
import dynamic from "next/dynamic";
import { useWindowDimensions } from "~/utils/windowDimensions";
import Data from "~/data/graph.json";

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
