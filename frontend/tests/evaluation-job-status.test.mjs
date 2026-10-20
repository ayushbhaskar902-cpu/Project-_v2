import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  EvaluationJobStatus,
  fetchEvaluationJobs,
} from '../src/screens/research/evaluation-job-status.ts'

const job = (overrides = {}) => ({
  jobId: 42,
  status: 'RESULT_BACKFILL',
  backfillStatus: 'PENDING',
  backfillErrorClass: null,
  lastError: null,
  nextRetryAt: null,
  ...overrides,
})

function render(jobs) {
  return renderToStaticMarkup(
    React.createElement(EvaluationJobStatus, { jobs, onRefresh: () => {} }),
  )
}

describe('evaluation job status display', () => {
  it('shows jobs waiting for backfill', () => {
    assert.match(render([job()]), /Waiting for backfill/)
  })

  it('distinguishes active backfill from waiting', () => {
    assert.match(render([job({ backfillStatus: 'IN_PROGRESS' })]), /Backfill in progress/)
  })

  it('shows scheduled retry and available timing', () => {
    const html = render([
      job({ backfillStatus: 'RETRY_SCHEDULED', nextRetryAt: '2030-01-02T03:04:00Z' }),
    ])
    assert.match(html, /Retry scheduled at/)
  })

  it('identifies incomplete artifacts without presenting readiness', () => {
    const html = render([job({ backfillErrorClass: 'INCOMPLETE_ARTIFACTS' })])
    assert.match(html, /Artifacts incomplete/)
    assert.doesNotMatch(html, /Verified results ready for evaluation/)
  })

  it('shows terminal failures as failures, never success', () => {
    const html = render([
      job({ status: 'FAILED', backfillStatus: 'TERMINAL_FAILURE', lastError: 'permanent error' }),
    ])
    assert.match(html, /Terminal failure — permanent error/)
    assert.match(html, /0 verified and ready for evaluation; 0 evaluated/)
    assert.doesNotMatch(html, /Evaluation complete/)
    assert.match(html, /text-pnl-negative-text/)
  })

  it('shows human review as a distinct state', () => {
    assert.match(render([job({ backfillStatus: 'REVIEW_REQUIRED' })]), /Human review required/)
  })

  it('labels EVALUABLE only when the backend status explicitly grants readiness', () => {
    const html = render([job({ status: 'EVALUABLE', backfillStatus: 'COMPLETE' })])
    assert.match(html, /Verified results ready for evaluation/)
    assert.match(html, /1 verified and ready for evaluation/)
    assert.doesNotMatch(
      render([job({ status: 'RESULT_BACKFILL', backfillStatus: 'COMPLETE' })]),
      /1 verified and ready for evaluation/,
    )
  })

  it('renders an evaluated job as complete, not as pending', () => {
    const html = render([job({ status: 'EVALUATED' })])
    assert.match(html, /Evaluation complete/)
    assert.match(html, /0 verified and ready for evaluation; 1 evaluated/)
    assert.match(html, /text-pnl-positive-text/)
  })

  it('keeps refresh/poll requests read-only', async () => {
    const calls = []
    await fetchEvaluationJobs(42, async (...args) => {
      calls.push(args)
      return new Response('[]', { status: 200 })
    })
    assert.equal(calls.length, 1)
    assert.equal(calls[0][0], '/api/research/sessions/42/evaluation-jobs')
    assert.equal(calls[0][1].method, 'GET')
  })
})
