export default function Caption({caption = "", children}) {
    return (
        <figure className="mt-8 flex flex-col items-center">
            {children}
            <figcaption className="mt-2 italic">{caption}</figcaption>
        </figure>
    );
};