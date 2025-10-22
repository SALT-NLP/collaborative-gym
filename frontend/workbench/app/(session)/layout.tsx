import { GeistMono } from 'geist/font/mono'
import { GeistSans } from 'geist/font/sans'

import '@/app/globals.css'
import { Providers } from '@/components/providers'
import { TailwindIndicator } from '@/components/tailwind-indicator'
import { VerticalMenu } from '@/components/vertical-menu'
import { cn } from '@/lib/utils'

export const metadata = {
  title: {
    default: 'Collaborative Gym UI',
  },
  description: '',
}

export const viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' }
  ]
}

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div
      className={cn(
        'font-sans antialiased',
        GeistSans.variable,
        GeistMono.variable
      )}
    >
      <Providers
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        <div className="flex min-h-screen">
          {/* Sidebar */}
          <VerticalMenu />
          {/* Divider */}
          <div className="w-px bg-divider"></div>
          {/* Main Content */}
          <main className="flex-1 flex flex-col">
            {children}
          </main>
        </div>
        <TailwindIndicator />
      </Providers>
    </div>
  )
}
