import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "https://airdraw-studio.priyangshubala.chatgpt.site"),
  title: "AirDraw Studio — Draw with your hand",
  description: "A private, gesture-powered drawing canvas that runs entirely in your browser.",
  openGraph: { title: "AirDraw Studio", description: "Draw in the air. Create in the browser.", type: "website", images: [{ url: "/og.png", width: 1200, height: 630, alt: "AirDraw Studio neon light trails" }] },
  twitter: { card: "summary_large_image", title: "AirDraw Studio", description: "Draw in the air. Create in the browser.", images: ["/og.png"] },
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
