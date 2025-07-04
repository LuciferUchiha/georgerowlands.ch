"use client";
import React, { useState } from "react";

const YouTube = ({ href, caption }) => {
  const [isOpen, setIsOpen] = useState(false);
  console.log("Is Open:", isOpen);

  const openModal = (e) => {
    e.preventDefault();
    setIsOpen(true);
  };
  const closeModal = () => setIsOpen(false);

  const getYouTubeId = (url) => {
    const regExp =
      /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  };

  const videoId = getYouTubeId(href);

  return (
    <>
      <a
        href={href}
        onClick={openModal}
        className="text-red-500 underline cursor-pointer"
      >
        {caption}
      </a>

      {isOpen && videoId && (
        <div
          className="fixed inset-0 bg-grey bg-opacity-[0.95] opacity-[0.95] z-50 flex justify-center items-center"
          onClick={closeModal}
        >
          <div className="aspect-w-16 aspect-h-9 min-w-[320px] min-h-[180px] w-[1080px] h-[720px]">
            <iframe
              src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}
              title="YouTube video player"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </div>
        </div>
      )}
    </>
  );
};

export default YouTube;
