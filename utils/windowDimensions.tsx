"use client";
import {useState, useLayoutEffect, useEffect} from "react";

const getWindowDimensions = () => {
    if (typeof window === "undefined") {
        return {
            width: 0,
            height: 0,
        };
    }
    return {
        width: window.innerWidth,
        height: window.innerHeight,
    };
};


export const useWindowDimensions = () => {
    const [windowDimensions, setWindowDimensions] = useState(
        getWindowDimensions()
    );

    useEffect(() => {
        function handleResize() {
            setWindowDimensions(getWindowDimensions());
        }

        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return windowDimensions;
};