export default function Caption({
  caption = "",
  children,
}: {
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="mt-8 flex flex-col items-center w-4/5">
      {children}
      <figcaption className="mt-2 italic">{caption}</figcaption>
    </figure>
  );
}

export function MdxCaption({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <figure className="mt-8 flex flex-col items-center">{children}</figure>
  );
}

