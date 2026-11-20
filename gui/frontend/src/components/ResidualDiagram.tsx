import { useId } from 'react'
import type { ResidualVariant } from '../services/learning'

export default function ResidualDiagram({ variant, layers }: { variant: ResidualVariant; layers?: number }) {
  const id = useId().replace(/:/g, '')
  const residuals = variant === 'with-residuals'
  return <figure className="mt-5 rounded-xl border border-slate-800 bg-slate-950/70 p-4">
    <figcaption className="mb-3 text-sm font-medium text-slate-200">{residuals ? 'With residual connections' : 'Without residual connections'} · one decoder block{layers ? `, repeated ${layers} times` : ''}</figcaption>
    <div className="overflow-x-auto">
      <svg role="img" aria-labelledby={`${id}-title ${id}-description`} viewBox="0 0 790 215" className="min-w-[620px] w-full text-slate-400">
        <title id={`${id}-title`}>{residuals ? 'Attention and feed-forward transformations with two bypass additions' : 'Attention followed by feed-forward transformation without bypass additions'}</title>
        <desc id={`${id}-description`}>{residuals ? 'The input bypasses RMSNorm and attention, and is added to their output. That result bypasses the next RMSNorm and feed-forward network, and is added to their output.' : 'The hidden state passes through RMSNorm and attention, then RMSNorm and the feed-forward network. There are no bypasses or additions.'}</desc>
        <defs><marker id={`${id}-arrow`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="currentColor" /></marker></defs>
        <g fill="none" stroke="currentColor" strokeWidth="2" markerEnd={`url(#${id}-arrow)`}>
          <path d="M109 108H151" /><path d={residuals ? 'M326 108H350' : 'M326 108H412'} />
          {residuals && <path d="M382 108H412" />}
          <path d={residuals ? 'M586 108H610' : 'M586 108H679'} />
          {residuals && <path d="M642 108H679" />}
          {residuals && <g className="text-emerald-400"><path d="M126 108V33H366V89" /><path d="M397 108V180H626V127" /></g>}
        </g>
        <g fill="#0f172a" stroke="#334155" strokeWidth="1.5"><rect x="12" y="82" width="97" height="52" rx="8" /><rect x="155" y="67" width="171" height="82" rx="8" /><rect x="415" y="67" width="171" height="82" rx="8" /><rect x="683" y="82" width="96" height="52" rx="8" /></g>
        <g fill="#e2e8f0" fontSize="13" textAnchor="middle"><text x="60" y="104">Hidden</text><text x="60" y="122">state</text><text x="240" y="98">RMSNorm</text><text x="240" y="122">Causal attention</text><text x="500" y="98">RMSNorm</text><text x="500" y="122">SwiGLU FFN</text><text x="731" y="113">Output</text></g>
        {residuals && <><g fill="#052e23" stroke="#34d399" strokeWidth="1.5"><circle cx="366" cy="108" r="16" /><circle cx="626" cy="108" r="16" /></g><g fill="#6ee7b7" textAnchor="middle" fontSize="22"><text x="366" y="115">+</text><text x="626" y="115">+</text></g><g fill="#6ee7b7" fontSize="12" textAnchor="middle"><text x="245" y="24">Keep the incoming representation</text><text x="510" y="200">Keep the next incoming representation</text></g></>}
      </svg>
    </div>
    <p className="mt-3 text-xs leading-5 text-slate-400">Arrows carry token representations. Each + adds two vectors of the same size, element by element. Both variants keep the same normalizations and attention/FFN settings; the variant without residuals removes the two additions and their bypasses.</p>
  </figure>
}
