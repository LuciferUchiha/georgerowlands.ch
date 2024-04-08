import { useTheme } from "nextra-theme-docs";
import dynamic from "next/dynamic";
import { useWindowDimensions } from "~/utils/windowDimensions";
import Data from "~/data/graph.json";
import SpriteText from "three-spritetext";
import { useRouter } from "next/navigation";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
});

const Graph = () => {
  const router = useRouter();
  const { theme, systemTheme } = useTheme();
  const currentTheme = theme !== "system" ? theme : systemTheme;
  const background = currentTheme === "dark" ? "#111" : "#fff";
  const textColor = currentTheme === "dark" ? "#fff" : "#000";

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
      nodeColor={() => "#8000FF"}
      nodeLabel={(node) =>
        `<span style="color: ${textColor}">${node.name}</span>`
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

export default Graph;
