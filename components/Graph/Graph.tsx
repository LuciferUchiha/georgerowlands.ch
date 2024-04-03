import { useTheme } from "nextra-theme-docs";
import dynamic from "next/dynamic";
import { useWindowDimensions } from "~/utils/windowDimensions";
import Data from "~/data/graph.json";
import SpriteText from "three-spritetext";
import { useRef } from "react";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
});

const Graph = () => {
  const { theme, systemTheme } = useTheme();
  const currentTheme = theme !== "system" ? theme : systemTheme;
  const background = currentTheme === "dark" ? "#111" : "#fff";
  const textColor = currentTheme === "dark" ? "#fff" : "#000";
  const linkOpacity = currentTheme === "dark" ? 0.2 : 1;

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
      nodeThreeObject={node => {
        const sprite = new SpriteText(node.id);
        sprite.color = textColor;
        sprite.textHeight = 8;
        return sprite;
      }}
      nodeLabel={() => ''}
      linkOpacity={linkOpacity}
      onNodeHover={node => {
        // TODO go to page on click
        document.body.style.cursor = node ? "pointer" : null;
      }}
    />
  );
};

export default Graph;
