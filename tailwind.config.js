const colors = require("tailwindcss/colors");
// basically all of this is copied from the docs-theme which allows me to have the same settings as the nextra docs theme
// without having to use the nx prefix (mainly wanted this for the primary color amonst other things).
const makePrimaryColor =
    l =>
        ({opacityValue}) => {
            if (opacityValue === undefined) {
                return `hsl(var(--nextra-primary-hue) 100% ${l}%)`;
            }
            return `hsl(var(--nextra-primary-hue) 100% ${l}% / ${opacityValue})`;
        };

/** @type {import("tailwindcss").Config} */
module.exports = {
  content: [
    './pages/**/*.{js,jsx,ts,tsx,md,mdx}',
    './components/**/*.{js,jsx,ts,tsx,md,mdx}',
    './theme.config.tsx'
  ],
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
        colors: {
            transparent: "transparent",
            current: "currentColor",
            black: "#000",
            white: "#fff",
            gray: colors.gray,
            slate: colors.slate,
            neutral: colors.neutral,
            red: colors.red,
            orange: colors.orange,
            blue: colors.blue,
            yellow: colors.yellow,
            primary: {
                50: makePrimaryColor(97),
                100: makePrimaryColor(94),
                200: makePrimaryColor(86),
                300: makePrimaryColor(77),
                400: makePrimaryColor(66),
                500: makePrimaryColor(50),
                600: makePrimaryColor(45),
                700: makePrimaryColor(39),
                750: makePrimaryColor(35),
                800: makePrimaryColor(32),
                900: makePrimaryColor(24)
            }
        },
        extend: {
            colors: {
                dark: "#111"
            },
            animation: {
                progress: "progress 5s infinite",
                noise: "noise .2s infinite;"
            },
            keyframes: {
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
                }
            }

        }
    },
    darkMode: "class"
};