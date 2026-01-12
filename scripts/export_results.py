#!/usr/bin/env python3
"""
Multi-Model Orchestration - Results Export Script

Export session results to various formats (JSON, CSV, Markdown).
"""

import asyncio
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from council.persistence import CouncilDatabase
from loguru import logger


class ResultsExporter:
    """Export council session results."""
    
    def __init__(self, db_path: str = "data/council.db"):
        self.db = CouncilDatabase(db_path)
    
    async def export_to_json(
        self,
        output_path: str = "exports/sessions.json",
        limit: int = 100
    ):
        """Export sessions to JSON."""
        await self.db.initialize()
        
        sessions = await self.db.get_recent_sessions(limit=limit)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = [session.to_dict() for session in sessions]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(sessions)} sessions to {output_file}")
        return len(sessions)
    
    async def export_to_csv(
        self,
        output_path: str = "exports/sessions.csv",
        limit: int = 100
    ):
        """Export sessions to CSV."""
        await self.db.initialize()
        
        sessions = await self.db.get_recent_sessions(limit=limit)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Session ID',
                'Timestamp',
                'Question',
                'Consensus',
                'Num Responses',
                'Confidence'
            ])
            
            # Rows
            for session in sessions:
                data = session.to_dict()
                
                consensus_data = data.get('consensus', {})
                if isinstance(consensus_data, str):
                    consensus_text = consensus_data[:100]
                    confidence = 0.0
                else:
                    consensus_text = consensus_data.get('consensus', '')[:100]
                    confidence = consensus_data.get('confidence', 0.0)
                
                writer.writerow([
                    data['session_id'],
                    data['timestamp'],
                    data['question'][:200],
                    consensus_text,
                    len(data.get('responses', [])),
                    f"{confidence:.2f}" if confidence else "N/A"
                ])
        
        print(f"✅ Exported {len(sessions)} sessions to {output_file}")
        return len(sessions)
    
    async def export_to_markdown(
        self,
        output_path: str = "exports/sessions.md",
        limit: int = 20
    ):
        """Export sessions to Markdown."""
        await self.db.initialize()
        
        sessions = await self.db.get_recent_sessions(limit=limit)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Multi-Model Orchestration - Session History\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Sessions: {len(sessions)}\n\n")
            f.write("---\n\n")
            
            for i, session in enumerate(sessions, 1):
                data = session.to_dict()
                
                f.write(f"## Session {i}: {data['session_id'][:8]}\n\n")
                f.write(f"**Timestamp:** {data['timestamp']}\n\n")
                f.write(f"**Question:**\n> {data['question']}\n\n")
                
                # Consensus
                consensus_data = data.get('consensus', {})
                if isinstance(consensus_data, dict) and consensus_data.get('consensus'):
                    f.write(f"**Council Answer:**\n{consensus_data['consensus']}\n\n")
                    f.write(f"- Confidence: {consensus_data.get('confidence', 0):.1%}\n")
                    
                    if consensus_data.get('key_points'):
                        f.write(f"- Key Points:\n")
                        for point in consensus_data['key_points']:
                            f.write(f"  - {point}\n")
                    f.write("\n")
                
                # Individual responses
                responses = data.get('responses', [])
                if responses:
                    f.write(f"**Individual Responses ({len(responses)}):**\n\n")
                    for resp in responses:
                        f.write(f"- **{resp.get('model', 'Unknown')}** ({resp.get('role', 'N/A')}):\n")
                        f.write(f"  {resp.get('content', '')[:200]}...\n\n")
                
                f.write("---\n\n")
        
        print(f"✅ Exported {len(sessions)} sessions to {output_file}")
        return len(sessions)
    
    async def export_statistics(
        self,
        output_path: str = "exports/statistics.json"
    ):
        """Export database statistics."""
        await self.db.initialize()
        
        stats = await self.db.get_statistics()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Exported statistics to {output_file}")
        return stats


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export LLM Council results")
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'markdown', 'all'],
        default='all',
        help='Export format'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of sessions to export'
    )
    parser.add_argument(
        '--output-dir',
        default='exports',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📤 Multi-Model Orchestration - Results Export")
    print("="*60 + "\n")
    
    exporter = ResultsExporter()
    
    output_dir = Path(args.output_dir)
    
    if args.format in ['json', 'all']:
        await exporter.export_to_json(
            output_path=str(output_dir / "sessions.json"),
            limit=args.limit
        )
    
    if args.format in ['csv', 'all']:
        await exporter.export_to_csv(
            output_path=str(output_dir / "sessions.csv"),
            limit=args.limit
        )
    
    if args.format in ['markdown', 'all']:
        await exporter.export_to_markdown(
            output_path=str(output_dir / "sessions.md"),
            limit=min(args.limit, 20)  # Limit markdown to 20 for readability
        )
    
    # Always export statistics
    stats = await exporter.export_statistics(
        output_path=str(output_dir / "statistics.json")
    )
    
    print("\n" + "="*60)
    print("📊 Export Summary")
    print("="*60)
    print(f"\nTotal sessions in database: {stats.get('total_sessions', 0)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("\n✅ Export complete!\n")


if __name__ == "__main__":
    logger.remove()
    asyncio.run(main())
