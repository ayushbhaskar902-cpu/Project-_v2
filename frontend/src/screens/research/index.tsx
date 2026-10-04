/**
 * Research Studio: Autonomous Multi-LLM Alpha Creation Engine.
 *
 * Implements the live control dashboard for:
 * - Autonomous campaign lifecycle (generation, simulation, evaluation, reflection)
 * - Multi-axis leaderboard (Alphas, LLM Researchers, Prompt Strategies)
 * - Candidate factor stream with AST syntax validation feedback
 * - Closed-loop self-learning insights & operator efficacy attribution
 * - Low-correlation Alpha Pool curation targeting 50+ submission-grade alphas
 */

import {
  BrainIcon,
  CheckCircle2Icon,
  CpuIcon,
  LayersIcon,
  LightbulbIcon,
  PlayIcon,
  SparklesIcon,
  TrophyIcon,
  XCircleIcon,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Badge, Button, Page, PageHeader, Panel } from '@/ui/kit'

interface SessionSummary {
  id: number
  name: string
  prompt: string
  region: string
  universe: string
  delay: number
  status: string
  simulationBudget: number
  simulationsUsed: number
  learningFrequency: number
  currentGeneration: number
  totalCandidates: number
  validCandidates: number
  submissionGradeCandidates: number
  bestSharpe: number | null
}

interface Researcher {
  id: number
  modelRef: string
  label: string
  strategyFocus: string
  temperature: number
  totalGenerated: number
  totalValid: number
  avgSharpe: number | null
  bestSharpe: number | null
  enabled: boolean
}

interface AlphaRank {
  rank: number
  candidateId: number
  expression: string
  hypothesis: string
  modelLabel: string
  generation: number
  compositeScore: number
  fullSharpe: number
  afterCostSharpe: number
  oosSharpe: number
  sharpeDecay: number
  turnover: number
  fitness: number
  isSubmissionGrade: boolean
}

interface ResearcherRank {
  rank: number
  researcherId: number
  label: string
  modelRef: string
  strategyFocus: string
  totalGenerated: number
  totalValid: number
  validRate: number
  avgSharpe: number
  bestSharpe: number
  submissionGradeCount: number
  topDecileRate: number
}

interface CandidateItem {
  id: number
  generation: number
  hypothesis: string
  expression: string
  operatorCount: number
  fieldCount: number
  isValid: boolean
  validationError: string | null
  retryCount: number
  status: string
  fullSharpe: number | null
  compositeScore: number | null
  isSubmissionGrade: boolean
}

interface InsightItem {
  id: number
  generation: number
  insightType: string
  summary: string
  rules: Record<string, unknown>
  evidence: Record<string, unknown>
  batchAvgSharpe: number | null
  batchBestSharpe: number | null
}

interface PoolResult {
  poolSize: number
  combinedSharpe: number
  combinedReturns: number
  combinedTurnover: number
  combinedDrawdown: number
  maxCorrelation: number
  avgCorrelation: number
  members: Array<{
    candidateId: number
    expression: string
    sharpe: number
    fitness: number
    turnover: number
    compositeScore: number
    weight: number
  }>
  correlationMatrix: number[][]
}

export function ResearchStudioScreen() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null)
  const [researchers, setResearchers] = useState<Researcher[]>([])
  const [alphas, setAlphas] = useState<AlphaRank[]>([])
  const [researcherRanks, setResearcherRanks] = useState<ResearcherRank[]>([])
  const [candidates, setCandidates] = useState<CandidateItem[]>([])
  const [insights, setInsights] = useState<InsightItem[]>([])
  const [poolResult, setPoolResult] = useState<PoolResult | null>(null)

  const [activeTab, setActiveTab] = useState<'leaderboards' | 'candidates' | 'learning' | 'pool'>(
    'leaderboards',
  )
  const [isGenerating, setIsGenerating] = useState(false)
  const [isSimulating, setIsSimulating] = useState(false)
  const [isLearning, setIsLearning] = useState(false)
  const [isCurating, setIsCurating] = useState(false)
  const [maxCorr, setMaxCorr] = useState(0.6)
  const [targetPoolSize, setTargetPoolSize] = useState(50)

  // Fetch sessions on load
  const loadSessions = useCallback(async () => {
    try {
      const res = await fetch('/api/research/sessions', {
        headers: { 'sec-fetch-site': 'same-origin' },
      })
      if (res.ok) {
        const data = await res.json()
        setSessions(data)
        if (data.length > 0 && selectedSessionId === null) {
          setSelectedSessionId(data[0].id)
        }
      }
    } catch (err) {
      console.error('Failed to load sessions:', err)
    }
  }, [selectedSessionId])

  // Fetch session details when selection changes
  const loadSessionData = useCallback(async (sid: number) => {
    try {
      const [rRes, cRes, aRes, rrRes, iRes] = await Promise.all([
        fetch(`/api/research/sessions/${sid}/researchers`, {
          headers: { 'sec-fetch-site': 'same-origin' },
        }),
        fetch(`/api/research/sessions/${sid}/candidates?limit=50`, {
          headers: { 'sec-fetch-site': 'same-origin' },
        }),
        fetch(`/api/research/leaderboards/alphas?session_id=${sid}&limit=50`, {
          headers: { 'sec-fetch-site': 'same-origin' },
        }),
        fetch(`/api/research/leaderboards/researchers?session_id=${sid}`, {
          headers: { 'sec-fetch-site': 'same-origin' },
        }),
        fetch(`/api/research/sessions/${sid}/learning/insights`, {
          headers: { 'sec-fetch-site': 'same-origin' },
        }),
      ])

      if (rRes.ok) setResearchers(await rRes.json())
      if (cRes.ok) setCandidates(await cRes.json())
      if (aRes.ok) setAlphas(await aRes.json())
      if (rrRes.ok) setResearcherRanks(await rrRes.json())
      if (iRes.ok) setInsights(await iRes.json())
    } catch (err) {
      console.error('Failed to load session data:', err)
    }
  }, [])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    if (selectedSessionId !== null) {
      loadSessionData(selectedSessionId)
    }
  }, [selectedSessionId, loadSessionData])

  // Actions
  const handleTriggerGeneration = async () => {
    if (!selectedSessionId) return
    setIsGenerating(true)
    try {
      const res = await fetch(`/api/research/sessions/${selectedSessionId}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'sec-fetch-site': 'same-origin',
          'x-harness-client': '1',
        },
        body: JSON.stringify({ total_candidates: 80 }),
      })
      if (res.ok) {
        const data = await res.json()
        const count = data.validCandidatesGenerated ?? 0
        toast.success(`Dispatched round: ${count} candidates generated`)
        await loadSessionData(selectedSessionId)
        await loadSessions()
      } else {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }))
        toast.error('Dispatch failed', { description: String(err.detail || 'Server error') })
      }
    } catch (err) {
      toast.error('Dispatch error', { description: String(err) })
    } finally {
      setIsGenerating(false)
    }
  }

  const handleSimulateCandidates = async () => {
    if (!selectedSessionId) return
    setIsSimulating(true)
    try {
      // 1. Enqueue to BatchEngine for live WorldQuant BRAIN simulation
      await fetch(`/api/research/sessions/${selectedSessionId}/enqueue`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'sec-fetch-site': 'same-origin',
          'x-harness-client': '1',
        },
        body: JSON.stringify({}),
      })

      // 2. Evaluate candidates and generate full 10-year scorecards
      const res = await fetch(`/api/research/sessions/${selectedSessionId}/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'sec-fetch-site': 'same-origin',
          'x-harness-client': '1',
        },
        body: JSON.stringify({ cost_bps: 1.5 }),
      })

      if (res.ok) {
        const data = await res.json()
        if (data.status === 'pending_simulations') {
          toast.info('Simulations Dispatched to BRAIN', {
            description: data.message || `${data.pendingSimulations} simulations running on WorldQuant BRAIN.`,
          })
        } else if (data.evaluatedCount > 0) {
          toast.success(
            `BRAIN Backtest Evaluation Complete: ${data.evaluatedCount} alphas (${data.submissionGradeCount} submissible)`,
          )
        } else {
          toast.info('Evaluation Status', {
            description: data.message || 'No completed BRAIN simulations to evaluate yet.',
          })
        }
        await loadSessionData(selectedSessionId)
        await loadSessions()
      } else {
        const err = await res.json().catch(() => ({ detail: 'Simulation failed' }))
        toast.error('Simulation failed', { description: String(err.detail || 'Server error') })
      }
    } catch (err) {
      toast.error('Simulation error', { description: String(err) })
    } finally {
      setIsSimulating(false)
    }
  }

  const handleTriggerLearningCycle = async () => {
    if (!selectedSessionId) return
    setIsLearning(true)
    try {
      const res = await fetch(`/api/research/sessions/${selectedSessionId}/learning/cycle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'sec-fetch-site': 'same-origin',
          'x-harness-client': '1',
        },
      })
      if (res.ok) {
        const data = await res.json()
        toast.success(`Learning cycle complete (advanced to Gen ${data.newGeneration ?? 'next'})`)
        await loadSessionData(selectedSessionId)
        await loadSessions()
      } else {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }))
        toast.error('Learning cycle failed', { description: String(err.detail || 'Server error') })
      }
    } catch (err) {
      toast.error('Learning cycle error', { description: String(err) })
    } finally {
      setIsLearning(false)
    }
  }

  const handleCuratePool = async () => {
    if (!selectedSessionId) return
    setIsCurating(true)
    try {
      const res = await fetch('/api/research/pool/curate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'sec-fetch-site': 'same-origin',
          'x-harness-client': '1',
        },
        body: JSON.stringify({
          session_id: selectedSessionId,
          target_pool_size: targetPoolSize,
          max_pairwise_correlation: maxCorr,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        setPoolResult(data)
        toast.success(`Pool curated: ${data.poolSize} uncorrelated alphas selected`)
      } else {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }))
        toast.error('Pool curation failed', { description: String(err.detail || 'Server error') })
      }
    } catch (err) {
      toast.error('Pool curation error', { description: String(err) })
    } finally {
      setIsCurating(false)
    }
  }

  const currentSession = sessions.find((s) => s.id === selectedSessionId)

  return (
    <Page>
      <PageHeader
        title={
          <div className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/20 text-primary">
              <SparklesIcon className="h-4 w-4" />
            </span>
            <span>Research Studio</span>
            <Badge tone="profit" className="ml-1 text-[10px]">
              AUTONOMOUS MULTI-LLM
            </Badge>
          </div>
        }
        description="Autonomous quantitative research laboratory. Heterogeneous LLM researcher agents collaborate, generate AST-validated Fast Expressions, evaluate 10-year simulation metrics, and self-learn optimal alpha strategies."
        actions={
          <div className="flex items-center gap-2">
            {sessions.length > 0 && (
              <select
                value={selectedSessionId ?? ''}
                onChange={(e) => setSelectedSessionId(Number(e.target.value))}
                className="h-8 rounded border border-hairline bg-surface-2 px-2.5 py-1 text-caption font-medium text-ink focus:border-primary focus:outline-none"
              >
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} (Gen {s.currentGeneration})
                  </option>
                ))}
              </select>
            )}
            <Button
              variant="secondary"
              disabled={isSimulating || !selectedSessionId}
              onClick={handleSimulateCandidates}
            >
              <CpuIcon className="mr-1.5 h-3.5 w-3.5 text-status-running" />
              {isSimulating ? 'Simulating & Evaluating...' : 'Simulate Round'}
            </Button>
            <Button
              variant="secondary"
              disabled={isLearning || !selectedSessionId}
              onClick={handleTriggerLearningCycle}
            >
              <LightbulbIcon className="mr-1.5 h-3.5 w-3.5 text-status-warning" />
              {isLearning ? 'Reflecting...' : 'Run Learning Cycle'}
            </Button>
            <Button
              variant="primary"
              disabled={isGenerating || !selectedSessionId}
              onClick={handleTriggerGeneration}
            >
              <PlayIcon className="mr-1.5 h-3.5 w-3.5" />
              {isGenerating ? 'Generating (80 Sims)...' : 'Dispatch Round (80 Sims)'}
            </Button>
          </div>
        }
      />

      {/* Campaign Overview Banner */}
      {currentSession && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Panel className="p-3">
            <div className="text-[11px] font-medium text-ink-subtle uppercase tracking-wider">
              Campaign
            </div>
            <div className="mt-1 truncate text-body font-semibold text-ink">
              {currentSession.name}
            </div>
            <div className="text-[11px] text-ink-muted">
              {currentSession.region} / {currentSession.universe}
            </div>
          </Panel>
          <Panel className="p-3">
            <div className="text-[11px] font-medium text-ink-subtle uppercase tracking-wider">
              Generation
            </div>
            <div className="mt-1 text-headline font-bold text-ink">
              Gen {currentSession.currentGeneration}
            </div>
            <div className="text-[11px] text-ink-muted">
              Every {currentSession.learningFrequency} sims
            </div>
          </Panel>
          <Panel className="p-3">
            <div className="text-[11px] font-medium text-ink-subtle uppercase tracking-wider">
              Total Candidates
            </div>
            <div className="mt-1 text-headline font-bold text-ink">
              {currentSession.totalCandidates}
            </div>
            <div className="text-[11px] text-pnl-positive-text">
              {currentSession.validCandidates} AST valid
            </div>
          </Panel>
          <Panel className="p-3">
            <div className="text-[11px] font-medium text-ink-subtle uppercase tracking-wider">
              Submission Grade
            </div>
            <div className="mt-1 text-headline font-bold text-pnl-positive-text">
              {currentSession.submissionGradeCandidates}
            </div>
            <div className="text-[11px] text-ink-muted">Sharpe ≥ 1.25, Fitness ≥ 1.0</div>
          </Panel>
          <Panel className="p-3">
            <div className="text-[11px] font-medium text-ink-subtle uppercase tracking-wider">
              Peak Sharpe
            </div>
            <div className="mt-1 text-headline font-bold text-pnl-positive-text">
              {currentSession.bestSharpe ? currentSession.bestSharpe.toFixed(2) : '—'}
            </div>
            <div className="text-[11px] text-ink-muted">10-year After-Cost</div>
          </Panel>
          <Panel className="p-3">
            <div className="text-[11px] font-medium text-ink-subtle uppercase tracking-wider">
              Researchers
            </div>
            <div className="mt-1 text-headline font-bold text-ink">{researchers.length}</div>
            <div className="text-[11px] text-ink-muted">Parallel LLM agents</div>
          </Panel>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex border-b border-hairline">
        <button
          type="button"
          onClick={() => setActiveTab('leaderboards')}
          className={`px-4 py-2.5 text-caption font-semibold transition-colors border-b-2 -mb-px flex items-center gap-1.5 ${
            activeTab === 'leaderboards'
              ? 'border-primary text-ink'
              : 'border-transparent text-ink-subtle hover:text-ink'
          }`}
        >
          <TrophyIcon className="h-3.5 w-3.5" />
          Multi-Axis Leaderboards
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('candidates')}
          className={`px-4 py-2.5 text-caption font-semibold transition-colors border-b-2 -mb-px flex items-center gap-1.5 ${
            activeTab === 'candidates'
              ? 'border-primary text-ink'
              : 'border-transparent text-ink-subtle hover:text-ink'
          }`}
        >
          <CpuIcon className="h-3.5 w-3.5" />
          Candidate Stream & AST Validation
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('learning')}
          className={`px-4 py-2.5 text-caption font-semibold transition-colors border-b-2 -mb-px flex items-center gap-1.5 ${
            activeTab === 'learning'
              ? 'border-primary text-ink'
              : 'border-transparent text-ink-subtle hover:text-ink'
          }`}
        >
          <BrainIcon className="h-3.5 w-3.5" />
          Self-Learning & Memory ({insights.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('pool')}
          className={`px-4 py-2.5 text-caption font-semibold transition-colors border-b-2 -mb-px flex items-center gap-1.5 ${
            activeTab === 'pool'
              ? 'border-primary text-ink'
              : 'border-transparent text-ink-subtle hover:text-ink'
          }`}
        >
          <LayersIcon className="h-3.5 w-3.5" />
          Alpha Pool Optimizer (50+ Target)
        </button>
      </div>

      {/* Tab 1: Multi-Axis Leaderboard */}
      {activeTab === 'leaderboards' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Alpha Candidate Leaderboard"
            description="Ranked by composite score: 0.40 * AfterCostSharpe + 0.30 * Fitness + 0.20 * (1 - SharpeDecay) - 0.10 * Turnover."
          >
            {alphas.length === 0 ? (
              <div className="py-8 text-center text-caption text-ink-subtle">
                No simulated alphas yet. Click "Dispatch Round (80 Sims)" to begin automated
                research.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-caption border-collapse">
                  <thead>
                    <tr className="border-b border-hairline text-ink-subtle">
                      <th className="py-2 px-3">#</th>
                      <th className="py-2 px-3">Expression</th>
                      <th className="py-2 px-3">Model</th>
                      <th className="py-2 px-3 text-right">Sharpe</th>
                      <th className="py-2 px-3 text-right">OOS Sharpe</th>
                      <th className="py-2 px-3 text-right">Decay</th>
                      <th className="py-2 px-3 text-right">Turnover</th>
                      <th className="py-2 px-3 text-right">Fitness</th>
                      <th className="py-2 px-3 text-right font-semibold text-ink">Score</th>
                      <th className="py-2 px-3 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alphas.map((a) => (
                      <tr
                        key={a.candidateId}
                        className="border-b border-hairline/50 hover:bg-surface-2 transition-colors"
                      >
                        <td className="py-2.5 px-3 font-mono font-bold text-ink-muted">{a.rank}</td>
                        <td
                          className="py-2.5 px-3 font-mono text-[12px] max-w-md truncate text-ink"
                          title={a.expression}
                        >
                          {a.expression}
                          <div className="text-[10px] text-ink-subtle truncate">{a.hypothesis}</div>
                        </td>
                        <td className="py-2.5 px-3 text-[11px] text-ink-muted">{a.modelLabel}</td>
                        <td className="py-2.5 px-3 text-right font-mono text-pnl-positive-text font-semibold">
                          {a.fullSharpe.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-ink">
                          {a.oosSharpe.toFixed(2)}
                        </td>
                        <td
                          className={`py-2.5 px-3 text-right font-mono ${a.sharpeDecay > 0.3 ? 'text-pnl-negative-text' : 'text-ink-muted'}`}
                        >
                          {(a.sharpeDecay * 100).toFixed(0)}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-ink-muted">
                          {(a.turnover * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-ink">
                          {a.fitness.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono font-bold text-primary">
                          {a.compositeScore.toFixed(3)}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          {a.isSubmissionGrade ? (
                            <Badge tone="profit" className="text-[10px]">
                              SUBMISSIBLE
                            </Badge>
                          ) : (
                            <Badge tone="warn" className="text-[10px]">
                              EVALUATED
                            </Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          {/* Model / Researcher Leaderboard */}
          <Panel
            title="LLM Researcher Persona Leaderboard"
            description="Attribution across heterogeneous LLM agents: yield rate of submission-grade factors and average Sharpe."
          >
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {researcherRanks.map((r) => (
                <div
                  key={r.researcherId}
                  className="flex flex-col justify-between rounded-lg border border-hairline bg-surface-1 p-3.5"
                >
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] font-bold text-primary">
                        RANK #{r.rank}
                      </span>
                      <Badge
                        tone={r.submissionGradeCount > 0 ? 'profit' : 'neutral'}
                        className="text-[10px]"
                      >
                        {r.submissionGradeCount} Submissible
                      </Badge>
                    </div>
                    <div className="mt-1.5 text-body font-semibold text-ink">{r.label}</div>
                    <div className="text-[11px] text-ink-subtle">{r.modelRef}</div>
                    <div className="mt-2 text-[11px] text-ink-muted capitalize">
                      Focus: {r.strategyFocus}
                    </div>
                  </div>
                  <div className="mt-4 grid grid-cols-3 gap-2 border-t border-hairline pt-3 text-center">
                    <div>
                      <div className="text-[10px] text-ink-subtle">Generated</div>
                      <div className="font-mono text-caption font-semibold text-ink">
                        {r.totalGenerated}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-ink-subtle">Avg Sharpe</div>
                      <div className="font-mono text-caption font-semibold text-pnl-positive-text">
                        {r.avgSharpe.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-ink-subtle">Best Sharpe</div>
                      <div className="font-mono text-caption font-semibold text-pnl-positive-text">
                        {r.bestSharpe.toFixed(2)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}

      {/* Tab 2: Candidate Stream & AST Validation */}
      {activeTab === 'candidates' && (
        <Panel
          title="Generated Fast Expression Stream"
          description="Real-time candidate generation with offline Fast Expression AST grammar validation, operator counting (≤ 8), field counting (≤ 3), and automated syntax repair."
        >
          <div className="flex flex-col divide-y divide-hairline">
            {candidates.map((c) => (
              <div key={c.id} className="py-3.5 flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] text-ink-subtle">ID #{c.id}</span>
                    <span className="text-[11px] text-ink-muted">Gen {c.generation}</span>
                    {c.isValid ? (
                      <Badge tone="profit" className="text-[10px] flex items-center gap-1">
                        <CheckCircle2Icon className="h-3 w-3" />
                        VALID AST
                      </Badge>
                    ) : (
                      <Badge tone="loss" className="text-[10px] flex items-center gap-1">
                        <XCircleIcon className="h-3 w-3" />
                        INVALID
                      </Badge>
                    )}
                    {c.retryCount > 0 && (
                      <Badge tone="warn" className="text-[10px]">
                        Auto-Repaired ({c.retryCount} retry)
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-[11px] font-mono text-ink-subtle">
                    <span>
                      Operators: <b className="text-ink">{c.operatorCount}/8</b>
                    </span>
                    <span>
                      Fields: <b className="text-ink">{c.fieldCount}/3</b>
                    </span>
                    {c.fullSharpe && (
                      <span className="text-pnl-positive-text font-bold">
                        Sharpe: {c.fullSharpe.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
                <div className="font-mono text-caption text-ink bg-surface-2 p-2 rounded border border-hairline overflow-x-auto">
                  {c.expression}
                </div>
                <div className="text-[11px] text-ink-subtle">{c.hypothesis}</div>
                {c.validationError && (
                  <div className="text-[11px] text-pnl-negative-text font-mono">
                    Error: {c.validationError}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* Tab 3: Self-Learning & Memory */}
      {activeTab === 'learning' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Closed-Loop Quantitative Knowledge Base"
            description="Empirical patterns and strategy updates extracted automatically after each generation round."
          >
            {insights.length === 0 ? (
              <div className="py-8 text-center text-caption text-ink-subtle">
                No insights extracted yet. The self-learning loop triggers automatically every 80
                simulations.
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {insights.map((ins) => (
                  <div
                    key={ins.id}
                    className="rounded-lg border border-hairline bg-surface-1 p-4 flex flex-col gap-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge
                          tone={ins.insightType === 'winning_pattern' ? 'profit' : 'warn'}
                          className="uppercase text-[10px]"
                        >
                          {ins.insightType}
                        </Badge>
                        <span className="font-mono text-caption font-semibold text-ink">
                          Generation {ins.generation} Discovery
                        </span>
                      </div>
                      <div className="text-[11px] font-mono text-pnl-positive-text">
                        Best Sharpe: {ins.batchBestSharpe?.toFixed(2)} | Avg:{' '}
                        {ins.batchAvgSharpe?.toFixed(2)}
                      </div>
                    </div>
                    <div className="text-body text-ink">{ins.summary}</div>
                    {ins.rules && Object.keys(ins.rules).length > 0 && (
                      <div className="mt-1 rounded bg-surface-2 p-2.5 font-mono text-[11px] text-ink-muted">
                        <div className="text-[10px] text-primary font-bold uppercase mb-1">
                          Extracted Strategy Heuristics:
                        </div>
                        <pre className="text-wrap">{JSON.stringify(ins.rules, null, 2)}</pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>
      )}

      {/* Tab 4: Alpha Pool Optimizer */}
      {activeTab === 'pool' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Alpha Pool Curation & Combo Optimizer"
            description="Selects high-capacity, uncorrelated alphas from your evaluated library to target 50+ submission-grade alphas with low mutual correlation."
            actions={
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 text-caption text-ink-subtle">
                  <span>Max Pairwise |ρ|:</span>
                  <input
                    type="number"
                    min="0.2"
                    max="0.9"
                    step="0.05"
                    value={maxCorr}
                    onChange={(e) => setMaxCorr(parseFloat(e.target.value) || 0.6)}
                    className="w-16 rounded border border-hairline bg-surface-2 px-2 py-1 text-ink font-mono text-right"
                  />
                </div>
                <div className="flex items-center gap-2 text-caption text-ink-subtle">
                  <span>Target Size:</span>
                  <input
                    type="number"
                    min="10"
                    max="100"
                    step="10"
                    value={targetPoolSize}
                    onChange={(e) => setTargetPoolSize(parseInt(e.target.value, 10) || 50)}
                    className="w-16 rounded border border-hairline bg-surface-2 px-2 py-1 text-ink font-mono text-right"
                  />
                </div>
                <Button
                  variant="primary"
                  disabled={isCurating || !selectedSessionId}
                  onClick={handleCuratePool}
                >
                  <LayersIcon className="mr-1.5 h-3.5 w-3.5" />
                  {isCurating ? 'Optimizing...' : 'Curate Alpha Pool'}
                </Button>
              </div>
            }
          >
            {poolResult ? (
              <div className="flex flex-col gap-4">
                {/* Combined Pool Metrics */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded border border-hairline bg-surface-2 p-3">
                    <div className="text-[11px] text-ink-subtle">Combined Portfolio Sharpe</div>
                    <div className="mt-1 text-headline font-bold text-pnl-positive-text">
                      {poolResult.combinedSharpe.toFixed(2)}
                    </div>
                  </div>
                  <div className="rounded border border-hairline bg-surface-2 p-3">
                    <div className="text-[11px] text-ink-subtle">Curated Members</div>
                    <div className="mt-1 text-headline font-bold text-ink">
                      {poolResult.poolSize} / {targetPoolSize}
                    </div>
                  </div>
                  <div className="rounded border border-hairline bg-surface-2 p-3">
                    <div className="text-[11px] text-ink-subtle">Max Pairwise Correlation</div>
                    <div className="mt-1 text-headline font-bold text-ink">
                      {poolResult.maxCorrelation.toFixed(2)}
                    </div>
                  </div>
                  <div className="rounded border border-hairline bg-surface-2 p-3">
                    <div className="text-[11px] text-ink-subtle">Mean Pairwise Correlation</div>
                    <div className="mt-1 text-headline font-bold text-ink">
                      {poolResult.avgCorrelation.toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Selected Members Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-caption border-collapse">
                    <thead>
                      <tr className="border-b border-hairline text-ink-subtle">
                        <th className="py-2 px-3">#</th>
                        <th className="py-2 px-3">Expression</th>
                        <th className="py-2 px-3 text-right">Individual Sharpe</th>
                        <th className="py-2 px-3 text-right">Fitness</th>
                        <th className="py-2 px-3 text-right">Turnover</th>
                        <th className="py-2 px-3 text-right">Portfolio Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {poolResult.members.map((m, idx) => (
                        <tr
                          key={m.candidateId}
                          className="border-b border-hairline/50 hover:bg-surface-2"
                        >
                          <td className="py-2 px-3 font-mono text-ink-muted">{idx + 1}</td>
                          <td className="py-2 px-3 font-mono text-ink text-[12px]">
                            {m.expression}
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-pnl-positive-text font-bold">
                            {m.sharpe.toFixed(2)}
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-ink">
                            {m.fitness.toFixed(2)}
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-ink-muted">
                            {(m.turnover * 100).toFixed(1)}%
                          </td>
                          <td className="py-2 px-3 text-right font-mono font-semibold text-primary">
                            {(m.weight * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-caption text-ink-subtle">
                Click "Curate Alpha Pool" above to run greedy correlation clustering across all
                simulated alphas.
              </div>
            )}
          </Panel>
        </div>
      )}
    </Page>
  )
}

export default ResearchStudioScreen
