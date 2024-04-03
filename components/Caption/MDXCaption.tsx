export default function MDXCaption({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <figure className="mt-8 flex flex-col items-center">{children}</figure>
  );
}
