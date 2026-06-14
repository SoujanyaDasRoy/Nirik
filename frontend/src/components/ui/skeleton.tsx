import { cn } from '@/lib/utils'

type SkeletonProps = {
  className?: string
  height?: number | string
  width?: number | string
  radius?: string
}

export function Skeleton({ className, height, width, radius = 'md' }: SkeletonProps) {
  const baseClasses = 'animate-pulse bg-muted/50 rounded'
  const radiusClasses = {
    none: 'rounded-none',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    xl: 'rounded-xl',
    '2xl': 'rounded-2xl',
  }[radius] || 'rounded-md'

  return (
    <div
      className={cn(baseClasses, radiusClasses, className)}
      style={{
        height: typeof height === 'number' ? `${height}px` : height,
        width: typeof width === 'number' ? `${width}px` : width,
      }}
    />
  )
}