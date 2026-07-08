import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KonthoProhori | Bangla Deepfake Audio Detection",
  description: "Bangla voice-forensics screening for suspicious audio clips.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

