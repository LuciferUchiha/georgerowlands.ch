import type { Config } from "tailwindcss"

type OpacityFunction = (params: { opacityValue: any }) => string;

// Define the makePrimaryColor function
const makePrimaryColor = (l: number): OpacityFunction => {
    return ({ opacityValue }) => {
        if (opacityValue === undefined) {
            return `hsl(var(--nextra-primary-hue) 100% ${l}%)`;
        }
        return `hsl(var(--nextra-primary-hue) 100% ${l}% / ${opacityValue})`;
    };
};

const config = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{js,jsx,ts,tsx,md,mdx}',
    './components/**/*.{js,jsx,ts,tsx,md,mdx}',
    './theme.config.tsx'
  ],
  prefix: "",
  theme: {
    screens: {
      sm: "640px",
      md: "768px",
      lg: "1024px",
      xl: "1280px",
      "2xl": "1536px"
    },
    fontSize: {
      xs: ".75rem",
      sm: ".875rem",
      base: "1rem",
      lg: "1.125rem",
      xl: "1.25rem",
      "2xl": "1.5rem",
      "3xl": "1.875rem",
      "4xl": "2.25rem",
      "5xl": "3rem",
      "6xl": "4rem"
    },
    letterSpacing: {
      tight: "-0.015em"
    },
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        dark: "#111",
        primary: {
          50: makePrimaryColor(97)({ opacityValue: undefined }),
          100: makePrimaryColor(94)({ opacityValue: undefined }),
          200: makePrimaryColor(86)({ opacityValue: undefined }),
          300: makePrimaryColor(77)({ opacityValue: undefined }),
          400: makePrimaryColor(66)({ opacityValue: undefined }),
          500: makePrimaryColor(50)({ opacityValue: undefined }),
          600: makePrimaryColor(45)({ opacityValue: undefined }),
          700: makePrimaryColor(39)({ opacityValue: undefined }),
          750: makePrimaryColor(35)({ opacityValue: undefined }),
          800: makePrimaryColor(32)({ opacityValue: undefined }),
          900: makePrimaryColor(24)({ opacityValue: undefined }),
        },
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        progress: {
          "0%": {width: "0%"},
          "100%": {width: "100%"}
        },
        noise: {
          "0%": {transform: "translate(0,0)"},
          "10%": {transform: "translate(-5%,-5%)"},
          "20%": {transform: "translate(-10%,5%)"},
          "30%": {transform: "translate(5%,-10%)"},
          "40%": {transform: "translate(-5%,15%)"},
          "50%": {transform: "translate(-10%,5%)"},
          "60%": {transform: "translate(15%,0)"},
          "70%": {transform: "translate(0,10%)"},
          "80%": {transform: "translate(-15%,0)"},
          "90%": {transform: "translate(10%,5%)"},
          "100%": {transform: "translate(5%,0)"}
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        progress: "progress 5s infinite",
        noise: "noise .2s infinite;"
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config

export default config