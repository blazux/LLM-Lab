import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import type { MetricPoint } from '../types'

interface Props {
  metrics: MetricPoint[]
}

const tooltipStyle = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 8,
  color: '#e2e8f0',
  fontSize: 12,
}

export default function MetricsPanel({ metrics }: Props) {
  if (metrics.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-600 text-sm">
        Waiting for metrics…
      </div>
    )
  }

  const lrData = metrics.filter((m) => m.lr !== undefined)

  return (
    <div className="space-y-6">
      {/* Loss chart */}
      <div>
        <div className="text-xs text-slate-500 mb-2 font-medium uppercase tracking-wider">Training Loss</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={metrics} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10, fill: '#475569' }}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#475569' }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => `Step ${v}`} />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#22c55e' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* LR chart */}
      {lrData.length > 0 && (
        <div>
          <div className="text-xs text-slate-500 mb-2 font-medium uppercase tracking-wider">Learning Rate</div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={lrData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="step"
                tick={{ fontSize: 10, fill: '#475569' }}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: '#475569' }}
                tickLine={false}
                axisLine={false}
                width={52}
                tickFormatter={(v: number) => v.toExponential(1)}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(v) => `Step ${v}`}
                formatter={(v) => [typeof v === 'number' ? v.toExponential(2) : String(v), 'lr']}
              />
              <Line
                type="monotone"
                dataKey="lr"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#3b82f6' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
