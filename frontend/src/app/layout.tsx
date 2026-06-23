import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";
import FetchInterceptor from "./components/FetchInterceptor";

const sansFont = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const monoFont = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});


export const metadata: Metadata = {
  title: "TB Diagnostic Assistant | AI-Powered Clinical Screening",
  description:
    "Deep learning-powered tuberculosis screening platform with DICOM support, Grad-CAM explainability, longitudinal tracking, and FHIR integration for medical professionals.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${sansFont.variable} ${monoFont.variable} h-full`}
    >
      <body className="min-h-full antialiased flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          forcedTheme="dark"
          enableSystem={false}
          disableTransitionOnChange={false}
        >
          <TooltipProvider delay={300}>
            <FetchInterceptor />
            {children}
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
