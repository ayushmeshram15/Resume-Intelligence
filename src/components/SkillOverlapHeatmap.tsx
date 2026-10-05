import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { RankedCandidate } from '../nlp/engine';

interface SkillOverlapHeatmapProps {
  candidates: RankedCandidate[];
  jdSkills: string[];
}

type SortMode = 'common-first' | 'unique-first' | 'jd-first';
type SkillFilterMode = 'all' | 'jd-only';

interface SkillColumnStat {
  skill: string;
  isJdSkill: boolean;
  count: number;
  frequencyRatio: number; // 0.0 to 1.0
  holders: string[];
}

interface HoveredCellInfo {
  candidateName: string;
  candidateRank: number;
  skill: string;
  hasSkill: boolean;
  isJdSkill: boolean;
  poolCount: number;
  totalCandidates: number;
  rarityLabel: string;
}

export const SkillOverlapHeatmap: React.FC<SkillOverlapHeatmapProps> = ({
  candidates,
  jdSkills,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>('common-first');
  const [filterMode, setFilterMode] = useState<SkillFilterMode>('all');
  const [hoveredCell, setHoveredCell] = useState<HoveredCellInfo | null>(null);

  const totalCandidates = candidates.length;

  // Build normalized skill statistics across the entire applicant pool + JD skills
  const skillStats: SkillColumnStat[] = useMemo(() => {
    const jdLowerSet = new Set(jdSkills.map((s) => s.toLowerCase().trim()));
    const canonicalMap = new Map<string, string>();

    for (const s of jdSkills) {
      canonicalMap.set(s.toLowerCase().trim(), s);
    }
    for (const cand of candidates) {
      for (const s of cand.skills) {
        const key = s.toLowerCase().trim();
        if (!canonicalMap.has(key)) {
          canonicalMap.set(key, s);
        }
      }
    }

    const stats: SkillColumnStat[] = [];
    for (const [lowerKey, displaySkill] of canonicalMap.entries()) {
      const isJdSkill = jdLowerSet.has(lowerKey);
      if (filterMode === 'jd-only' && !isJdSkill) {
        continue;
      }

      const holders: string[] = [];
      for (const cand of candidates) {
        const candLower = new Set(cand.skills.map((cs) => cs.toLowerCase().trim()));
        if (candLower.has(lowerKey)) {
          holders.push(cand.name);
        }
      }

      const count = holders.length;
      const frequencyRatio = totalCandidates > 0 ? count / totalCandidates : 0;

      stats.push({
        skill: displaySkill,
        isJdSkill,
        count,
        frequencyRatio,
        holders,
      });
    }

    stats.sort((a, b) => {
      if (sortMode === 'common-first') {
        if (b.count !== a.count) return b.count - a.count;
        if (a.isJdSkill !== b.isJdSkill) return a.isJdSkill ? -1 : 1;
        return a.skill.localeCompare(b.skill);
      }
      if (sortMode === 'unique-first') {
        // Put skills held by >0 candidates ordered rarest (1) to most common, then 0-count unmet JD skills
        const aRank = a.count === 0 ? 9999 : a.count;
        const bRank = b.count === 0 ? 9999 : b.count;
        if (aRank !== bRank) return aRank - bRank;
        return a.skill.localeCompare(b.skill);
      }
      // 'jd-first'
      if (a.isJdSkill !== b.isJdSkill) return a.isJdSkill ? -1 : 1;
      if (b.count !== a.count) return b.count - a.count;
      return a.skill.localeCompare(b.skill);
    });

    return stats;
  }, [candidates, jdSkills, sortMode, filterMode, totalCandidates]);

  // Identify Most Common vs. Most Unique (1 holder) vs. Unmet JD skills (0 holders)
  const { mostCommonSkills, uniquePoolSkills, unmetJdSkills } = useMemo(() => {
    const present = skillStats.filter((s) => s.count > 0);
    const maxCount = present.length > 0 ? Math.max(...present.map((s) => s.count)) : 0;

    const common = present
      .filter((s) => s.count === maxCount && maxCount > 0)
      .map((s) => `${s.skill} (${s.count}/${totalCandidates})`);

    const unique = present
      .filter((s) => s.count === 1)
      .map((s) => `${s.skill} (${s.holders[0]})`);

    const unmet = skillStats
      .filter((s) => s.isJdSkill && s.count === 0)
      .map((s) => s.skill);

    return {
      mostCommonSkills: common,
      uniquePoolSkills: unique,
      unmetJdSkills: unmet,
    };
  }, [skillStats, totalCandidates]);

  // Render D3 Heatmap
  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl || candidates.length === 0 || skillStats.length === 0) return;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const margin = { top: 120, right: 32, bottom: 48, left: 190 };
    const cellWidth = Math.max(46, Math.min(68, Math.floor(860 / Math.max(1, skillStats.length))));
    const cellHeight = 38;

    const innerWidth = skillStats.length * cellWidth;
    const innerHeight = candidates.length * cellHeight;

    const totalWidth = margin.left + innerWidth + margin.right;
    const totalHeight = margin.top + innerHeight + margin.bottom;

    svg
      .attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`)
      .attr('width', totalWidth)
      .attr('height', totalHeight);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xNames = skillStats.map((s) => s.skill);
    const yIds = candidates.map((c) => c.id);

    const xScale = d3
      .scaleBand<string>()
      .domain(xNames)
      .range([0, innerWidth])
      .paddingInner(0.08);

    const yScale = d3
      .scaleBand<string>()
      .domain(yIds)
      .range([0, innerHeight])
      .paddingInner(0.12);

    // Color scale for common skills (slate-blue intensity by pool frequency)
    const commonColorScale = d3
      .scaleSequential(d3.interpolateYlGnBu)
      .domain([0, Math.max(1, totalCandidates)]);

    // Column Headers: Rotated Skill Labels + Top Pool Count Readout
    const colHeaderGroup = g
      .selectAll('.col-header')
      .data(skillStats)
      .enter()
      .append('g')
      .attr('class', 'col-header')
      .attr(
        'transform',
        (d) => `translate(${(xScale(d.skill) ?? 0) + xScale.bandwidth() / 2}, -28)`
      );

    // Rotated Skill Name
    colHeaderGroup
      .append('text')
      .attr('transform', 'rotate(-40)')
      .attr('text-anchor', 'start')
      .attr('dx', 4)
      .attr('dy', 2)
      .attr('font-size', '11px')
      .attr('font-family', 'Plus Jakarta Sans, sans-serif')
      .attr('font-weight', (d) => (d.isJdSkill ? '700' : '500'))
      .attr('fill', (d) => (d.isJdSkill ? '#0F172A' : '#475569'))
      .text((d) => (d.isJdSkill ? `${d.skill} *` : d.skill));

    // Pool Frequency Count Badge right above each column (tabular-nums)
    colHeaderGroup
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('y', 18)
      .attr('font-size', '10px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-weight', '600')
      .attr('fill', (d) => {
        if (d.count === 1) return '#B45309'; // Unique skill highlight
        if (d.count === 0) return '#DC2626'; // Missing across all
        return '#334155';
      })
      .text((d) => `${d.count}/${totalCandidates}`);

    // Row Headers: Candidate Rank & Name
    const rowHeaderGroup = g
      .selectAll('.row-header')
      .data(candidates)
      .enter()
      .append('g')
      .attr('class', 'row-header')
      .attr(
        'transform',
        (d) => `translate(-12, ${(yScale(d.id) ?? 0) + yScale.bandwidth() / 2})`
      );

    rowHeaderGroup
      .append('text')
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '11px')
      .attr('font-family', 'Plus Jakarta Sans, sans-serif')
      .attr('font-weight', '600')
      .attr('fill', '#0F172A')
      .text((d) => {
        const label = `#${d.rank} ${d.name}`;
        return label.length > 24 ? `${label.slice(0, 22)}…` : label;
      });

    // Build flattened cell datum list
    interface CellDatum {
      candidate: RankedCandidate;
      stat: SkillColumnStat;
      hasSkill: boolean;
    }

    const cellsData: CellDatum[] = [];
    for (const cand of candidates) {
      const candSkillSet = new Set(cand.skills.map((s) => s.toLowerCase().trim()));
      for (const stat of skillStats) {
        cellsData.push({
          candidate: cand,
          stat,
          hasSkill: candSkillSet.has(stat.skill.toLowerCase().trim()),
        });
      }
    }

    // Draw Heatmap Cells
    const cellGroups = g
      .selectAll('.heatmap-cell')
      .data(cellsData)
      .enter()
      .append('g')
      .attr('class', 'heatmap-cell')
      .attr(
        'transform',
        (d) => `translate(${xScale(d.stat.skill) ?? 0}, ${yScale(d.candidate.id) ?? 0})`
      )
      .style('cursor', 'pointer')
      .on('mouseenter', function (_event, d) {
        d3.select(this).select('rect').attr('stroke', '#0F172A').attr('stroke-width', 2);

        let rarityLabel = 'Shared Pool Skill';
        if (d.stat.count === totalCandidates && totalCandidates > 1) {
          rarityLabel = 'Universal Pool Skill (100% of applicants)';
        } else if (d.stat.count === 1) {
          rarityLabel = 'Unique Skill (Only 1 applicant in pool)';
        } else if (d.stat.count === 0) {
          rarityLabel = 'Unmet JD Skill (0 applicants in pool)';
        }

        setHoveredCell({
          candidateName: d.candidate.name,
          candidateRank: d.candidate.rank,
          skill: d.stat.skill,
          hasSkill: d.hasSkill,
          isJdSkill: d.stat.isJdSkill,
          poolCount: d.stat.count,
          totalCandidates,
          rarityLabel,
        });
      })
      .on('mouseleave', function (_event, d) {
        d3.select(this)
          .select('rect')
          .attr('stroke', d.hasSkill && d.stat.count === 1 ? '#D97706' : '#E2E8F0')
          .attr('stroke-width', 1);
      });

    cellGroups
      .append('rect')
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('rx', 5)
      .attr('ry', 5)
      .attr('fill', (d) => {
        if (!d.hasSkill) {
          return d.stat.isJdSkill ? '#FFF1F2' : '#F8FAFC';
        }
        // Unique skill held by only 1 candidate in the pool -> warm amber highlight
        if (d.stat.count === 1) {
          return '#FEF3C7';
        }
        // Common/shared skill -> D3 sequential color intensity scaled by pool frequency
        return commonColorScale(Math.max(1, d.stat.count));
      })
      .attr('stroke', (d) => (d.hasSkill && d.stat.count === 1 ? '#D97706' : '#E2E8F0'))
      .attr('stroke-width', 1);

    // Explicit Accessible Symbol Inside Every Cell (Never rely on color alone)
    cellGroups
      .append('text')
      .attr('x', xScale.bandwidth() / 2)
      .attr('y', yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '11px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-weight', '600')
      .attr('fill', (d) => {
        if (!d.hasSkill) {
          return d.stat.isJdSkill ? '#FDA4AF' : '#CBD5E1';
        }
        if (d.stat.count === 1) {
          return '#92400E';
        }
        // Ensure high WCAG contrast on dark vs light sequential blues
        return d.stat.frequencyRatio >= 0.6 ? '#FFFFFF' : '#0F172A';
      })
      .text((d) => {
        if (!d.hasSkill) return '—';
        if (d.stat.count === 1) return '★';
        return '✓';
      });
  }, [candidates, skillStats, totalCandidates]);

  if (candidates.length === 0) {
    return null;
  }

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-6">
      {/* Header & Interactive Sort/Filter Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            4. Applicant Pool Skill Overlap & Rarity Heatmap (D3.js)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Compare common shared skills (<code className="font-mono">✓</code>) vs. unique single-candidate differentiators (<code className="font-mono">★</code>) and unmet JD requirements (<code className="font-mono">*</code>).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {/* Skill Scope Filter */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => setFilterMode('all')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                filterMode === 'all'
                  ? 'bg-white text-slate-900 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              All Pool Skills
            </button>
            <button
              type="button"
              onClick={() => setFilterMode('jd-only')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                filterMode === 'jd-only'
                  ? 'bg-white text-slate-900 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              JD Skills Only
            </button>
          </div>

          {/* Column Ordering Selector */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => setSortMode('common-first')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                sortMode === 'common-first'
                  ? 'bg-white text-slate-900 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Most Common First
            </button>
            <button
              type="button"
              onClick={() => setSortMode('unique-first')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                sortMode === 'unique-first'
                  ? 'bg-white text-slate-900 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Most Unique First
            </button>
            <button
              type="button"
              onClick={() => setSortMode('jd-first')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                sortMode === 'jd-first'
                  ? 'bg-white text-slate-900 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              JD Required First
            </button>
          </div>
        </div>
      </div>

      {/* Common vs. Unique Pool Summary Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
          <div className="font-semibold text-slate-900">
            Most Common Skills in Applicant Pool
          </div>
          <div className="font-mono text-slate-700 leading-relaxed">
            {mostCommonSkills.length > 0
              ? mostCommonSkills.slice(0, 6).join(' · ')
              : 'No skills detected'}
          </div>
        </div>

        <div className="p-3.5 rounded-lg bg-amber-50/70 border border-amber-200 space-y-1">
          <div className="font-semibold text-amber-900">
            Most Unique Skills (★ 1 Candidate Only)
          </div>
          <div className="font-mono text-amber-900 leading-relaxed">
            {uniquePoolSkills.length > 0
              ? uniquePoolSkills.slice(0, 6).join(' · ') +
                (uniquePoolSkills.length > 6 ? ` (+${uniquePoolSkills.length - 6} more)` : '')
              : 'None (all skills shared by ≥2 candidates)'}
          </div>
        </div>

        <div className="p-3.5 rounded-lg bg-rose-50/70 border border-rose-200 space-y-1">
          <div className="font-semibold text-rose-900">
            Unmet JD Skills (0 Candidates in Pool)
          </div>
          <div className="font-mono text-rose-900 leading-relaxed">
            {unmetJdSkills.length > 0
              ? unmetJdSkills.join(' · ')
              : 'None — Every JD skill is covered by at least 1 applicant'}
          </div>
        </div>
      </div>

      {/* D3 SVG Matrix Viewport */}
      {skillStats.length === 0 ? (
        <div className="p-8 text-center text-xs text-slate-500">
          No skills available to display for the current filter selection.
        </div>
      ) : (
        <div className="overflow-x-auto border border-slate-200 rounded-lg bg-white p-3">
          <svg ref={svgRef} className="block mx-auto" />
        </div>
      )}

      {/* Legend & Live Hover Cell Telemetry Readout */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2 text-xs text-slate-600">
        <div className="flex flex-wrap items-center gap-5">
          <span className="font-mono">
            <strong>✓</strong> Shared Skill (darker blue = more common)
          </span>
          <span className="font-mono text-amber-800">
            <strong>★</strong> Unique Differentiator (held by only 1 candidate)
          </span>
          <span className="font-mono text-slate-400">
            <strong>—</strong> Skill Absent
          </span>
          <span className="font-mono text-slate-800">
            <strong>*</strong> Required by JD
          </span>
        </div>

        <div className="font-mono text-xs text-slate-800 min-h-[20px]">
          {hoveredCell ? (
            <span>
              Rank #{hoveredCell.candidateRank} {hoveredCell.candidateName} ·{' '}
              <strong>{hoveredCell.skill}</strong>:{' '}
              {hoveredCell.hasSkill ? 'Present' : 'Missing'} ({hoveredCell.poolCount}/
              {hoveredCell.totalCandidates} in pool · {hoveredCell.rarityLabel})
            </span>
          ) : (
            <span className="text-slate-400">
              Hover over any cell in the heatmap to inspect candidate skill rarity
            </span>
          )}
        </div>
      </div>
    </section>
  );
};
