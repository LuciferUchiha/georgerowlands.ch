export default function Caption({
  caption = "",
  children,
}: {
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="mt-8 flex flex-col items-center">
      {children}
      <figcaption className="mt-2 italic">{caption}</figcaption>
    </figure>
  );
}

export function MDXCaption({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <figure className="mt-8 flex flex-col items-center">{children}</figure>
  );
}

