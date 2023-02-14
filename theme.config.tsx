import React from "react";
import {DocsThemeConfig} from "nextra-theme-docs";
import GiscusComments from "./components/giscus/giscus";
import BackToTop from "./components/backToTop/backToTop";


const config: DocsThemeConfig = {
        project: {
            link: "https://github.com/LuciferUchiha/nextra-garden"
        },
        docsRepositoryBase: "https://github.com/LuciferUchiha/nextra-garden/tree/main/",
        primaryHue: 270,
        nextThemes: {
            defaultTheme: "dark"
        },
        editLink: {
            text: "Edit this page on GitHub →"
        },
        feedback: {
            content: null // disable because now using Giscus
        },
        head: <link rel="icon" type="image/x-icon" href="logo.svg"/>,
        logo: <>
            <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
                 width="36px" height="36px" viewBox="0 0 128 128"
                 preserveAspectRatio="xMidYMid meet">

                <g transform="translate(0,128) scale(0.1,-0.1)">
                    <path fill="currentColor" d="M707 1214 c-187 -45 -463 -255 -529 -401 -69 -154 143 -163 395 -17
                                            44 25 81 43 83 41 6 -5 -54 -41 -141 -85 -134 -67 -344 -223 -400 -296 -15
                                            -19 -29 -53 -32 -74 -5 -38 -4 -40 32 -52 133 -44 298 71 483 337 110 157 112
                                            158 282 158 106 0 119 -2 168 -28 51 -27 86 -67 98 -115 19 -72 -46 -161 -159
                                            -217 -81 -40 -221 -78 -268 -73 l-31 3 53 80 c28 44 76 111 105 149 30 37 54
                                            74 54 82 0 53 -93 -45 -202 -213 l-73 -112 -41 -1 c-25 0 -47 -6 -54 -15 -11
                                            -13 -9 -20 13 -43 15 -16 27 -32 27 -36 0 -5 -21 -51 -47 -102 -26 -51 -45
                                            -96 -41 -99 17 -18 49 13 92 90 26 47 49 85 51 85 2 0 61 -31 132 -69 150 -81
                                            176 -91 171 -65 -2 11 -53 45 -133 88 -169 92 -156 82 -143 111 8 17 22 26 47
                                            30 227 36 342 78 427 157 52 48 74 90 74 142 0 145 -198 254 -394 217 -33 -6
                                            -61 -9 -63 -8 -2 2 13 25 32 50 19 25 35 48 35 51 0 21 -57 -3 -128 -54 -193
                                            -138 -406 -222 -450 -178 -38 38 7 117 135 238 134 127 313 222 423 224 49 1
                                            55 -1 58 -20 2 -12 -9 -35 -24 -53 -18 -22 -24 -35 -16 -43 23 -23 92 47 92
                                            95 0 55 -73 70 -193 41z m-156 -536 c-147 -213 -272 -320 -375 -322 -34 -1
                                            -41 2 -44 19 -7 53 141 187 340 308 61 37 115 67 120 67 5 0 -13 -33 -41 -72z"
                    />
                </g>
            </svg>
            <span className="ml-4 font-bold">George Rowlands</span></>,
        search: {
            // TODO I can replace the component that is shown for when nothing was found with an image of a dead tree
            placeholder: "Search in the garden..."
        },
        sidebar: {defaultMenuCollapseLevel: 2},
        toc: {
            title: "Table of contents"
        },
        main: ({children}) => {
            return (
                <>
                    {children}
                    <BackToTop/>
                    <GiscusComments/>
                </>
            );
        },
        footer: {
            text: () =>
                <div className="flex w-full flex-row items-center justify-between">
                    <ul>
                        <li><p className="font-bold">Socials</p></li>
                        <li><a href="https://www.linkedin.com/in/georgerowlands/" className="text-sm">Linkedin↗</a></li>
                        <li><a href="https://github.com/LuciferUchiha" className="text-sm">GitHub↗</a></li>
                        <li><a href="https://stackoverflow.com/users/10994912/george-r"
                               className="text-sm">StackOverflow↗</a></li>
                    </ul>
                    <a

                        href="https://www.buymeacoffee.com/georgerowlands" target="_blank"
                    >
                        <img
                            src="buyMeACoffeePurple.png"
                            alt="Buy Me A Coffee"
                            style={{height: "41px", width: "174px"}}
                        />
                    </a>
                </div>
        }
    }
;

export default config;
