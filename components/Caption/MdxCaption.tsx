export default function MdxCaption({children}) {
    return (
        <figure className="mt-8 flex flex-col items-center">
            {children}
        </figure>
    );
};