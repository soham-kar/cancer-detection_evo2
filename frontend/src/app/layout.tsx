import "~/styles/globals.css";

import { ClerkProvider } from "@clerk/nextjs";
import { type Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ActiveVariantProvider } from "~/contexts/active-variant";
import { ChatLayer } from "~/components/chat/chat-layer";

export const metadata: Metadata = {
  title: "HelixMind - Precision Genomic Variant Analysis",
  description: "AI-powered pathogenicity prediction for clinical research using Evo-2 and Llama-3",
  icons: [
    { rel: "icon", url: "/image_ulkl5pulkl5pulkl.png", type: "image/png" },
  ],
};

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <ClerkProvider
      appearance={{
        variables: {
          colorPrimary: '#de8246',
          colorText: '#3c4f3d',
          colorTextSecondary: '#3c4f3d99',
          colorBackground: '#ffffff',
          colorInputBackground: '#f4f7f5',
          colorInputText: '#3c4f3d',
          borderRadius: '0.75rem',
          fontFamily: 'Inter, system-ui, sans-serif',
        },
        elements: {
          card: 'shadow-xl border border-slate-200',
          headerTitle: 'text-[#3c4f3d] font-semibold',
          headerSubtitle: 'text-[#3c4f3d]/60',
          socialButtonsBlockButton: 'border-slate-200 hover:bg-slate-50',
          formButtonPrimary: 'bg-[#de8246] hover:bg-[#c97340] shadow-sm',
          formFieldInput: 'border-slate-200 focus:border-[#de8246] focus:ring-[#de8246]/20',
          navbarButton: 'text-[#3c4f3d] hover:bg-[#e9eeea]',
          profileSectionPrimaryButton: 'text-[#de8246] hover:text-[#c97340]',
          badge: 'bg-[#e9eeea] text-[#3c4f3d]',
          userButtonPopoverCard: 'shadow-xl border border-slate-200',
          userButtonPopoverActionButton: 'hover:bg-[#e9eeea]',
          footerActionLink: 'text-[#de8246] hover:text-[#c97340]',
        }
      }}
    >
      <ActiveVariantProvider>
        <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
          <body className="relative">
            {children}
            <ChatLayer />
          </body>
        </html>
      </ActiveVariantProvider>
    </ClerkProvider>
  );
}
