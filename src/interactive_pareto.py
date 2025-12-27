import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class InteractiveParetoExplorer:
    """Explorador interativo da Frente de Pareto."""
    
    def __init__(self, results_path: str = "data/output/moead_results.pkl"):
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Resultados não encontrados: {self.results_path}")
        
        with open(self.results_path, "rb") as f:
            self.results = pickle.load(f)
        
        self.pareto_front = self.results["pareto_front"]
        self.selected_solutions = []
        self.fig, (self.ax_plot, self.ax_info) = plt.subplots(1, 2, figsize=(16, 7))
        
    def _get_walk_distance(self, sol):
        """Calcula distância a pé de uma solução."""
        if hasattr(sol, 'edges') and sol.edges:
            return sum(e.get('distance_km', 0.0) for e in sol.edges if e.get('modo') == 'walk')
        return 0.0
    
    def _get_transfers(self, sol):
        """Calcula número de transbordos (mudanças de modo de transporte)."""
        if not hasattr(sol, 'edges') or not sol.edges:
            return 0
        
        transfers = 0
        prev_mode = None
        
        for e in sol.edges:
            mode = e.get('modo')
            # Conta transbordo quando muda de modo (exceto walk)
            if prev_mode is not None and mode != prev_mode:
                # Só conta se envolve transporte público (metro/bus)
                if prev_mode in ['metro', 'bus'] or mode in ['metro', 'bus']:
                    transfers += 1
            prev_mode = mode
        
        return transfers
    
    def _get_mode_breakdown(self, sol):
        """Retorna breakdown de modos."""
        if not hasattr(sol, 'edges') or not sol.edges:
            return {}
        
        totals = {'walk': {'time': 0.0, 'dist': 0.0, 'edges': 0},
                  'metro': {'time': 0.0, 'dist': 0.0, 'edges': 0},
                  'bus': {'time': 0.0, 'dist': 0.0, 'edges': 0}}
        
        for e in sol.edges:
            m = e.get('modo', 'walk')
            if m not in totals:
                totals[m] = {'time': 0.0, 'dist': 0.0, 'edges': 0}
            totals[m]['time'] += e.get('time_min', 0.0)
            totals[m]['dist'] += e.get('distance_km', 0.0)
            totals[m]['edges'] += 1
        
        return totals
    
    def _get_segments(self, sol):
        """Retorna segmentos contíguos por modo."""
        if not hasattr(sol, 'edges') or not sol.edges or len(sol.path) < 2:
            return []
        
        segments = []
        current_mode = None
        seg_start_idx = 0
        acc_time = 0.0
        acc_dist = 0.0
        
        for i, e in enumerate(sol.edges):
            mode = e.get('modo')
            time_min = e.get('time_min', 0.0)
            dist_km = e.get('distance_km', 0.0)
            
            if current_mode is None:
                current_mode = mode
            
            if mode != current_mode:
                segments.append((current_mode, sol.path[seg_start_idx], sol.path[i], acc_time, acc_dist))
                current_mode = mode
                seg_start_idx = i
                acc_time = 0.0
                acc_dist = 0.0
            
            acc_time += time_min
            acc_dist += dist_km
        
        segments.append((current_mode, sol.path[seg_start_idx], sol.path[-1], acc_time, acc_dist))
        return segments
    
    def _format_solution_info(self, sol, idx):
        """Formata informações de uma solução."""
        walk_dist = self._get_walk_distance(sol)
        transfers = self._get_transfers(sol)
        breakdown = self._get_mode_breakdown(sol)
        segments = self._get_segments(sol)
        
        info = f"================== SOLUÇÃO #{idx} ==================\n\n"
        info += f"Tempo total: {sol.time:.1f} min\n"
        info += f"CO₂ total: {sol.co2:.0f} g\n"
        info += f"Paragens: {len(sol.path)}\n"
        info += f"Transbordos: {transfers}\n"
        info += f"Distância a pé: {walk_dist:.2f} km\n\n"
        
        info += "Resumo por modo:\n"
        for m in ['walk', 'metro', 'bus']:
            if breakdown.get(m, {}).get('edges', 0) > 0:
                t = breakdown[m]['time']
                d = breakdown[m]['dist']
                e = breakdown[m]['edges']
                info += f"  {m.upper()}: {t:.1f} min | {d:.2f} km | {e} arestas\n"
        
        info += "\nCaminho:\n"
        for mode, a, b, tmin, dkm in segments:
            # Truncar nomes longos
            a_short = a[:15] + "..." if len(a) > 18 else a
            b_short = b[:15] + "..." if len(b) > 18 else b
            info += f"  {mode.upper()}: {a_short} → {b_short}\n"
            info += f"    {tmin:.1f} min | {dkm:.2f} km\n"
        
        return info
    
    def _on_pick(self, event):
        """Handler para clique em ponto."""
        if event.artist != self.scatter:
            return
        
        ind = event.ind[0]
        sol = self.pareto_front[ind]
        
        # Atualizar painel de informações
        self.ax_info.clear()
        self.ax_info.axis('off')
        info_text = self._format_solution_info(sol, ind)
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Destacar ponto selecionado
        self.scatter.set_edgecolors(['red' if i == ind else 'none' for i in range(len(self.pareto_front))])
        self.scatter.set_linewidths([2 if i == ind else 0 for i in range(len(self.pareto_front))])
        
        self.fig.canvas.draw()
    
    def _reset_selection(self, event):
        """Reset da seleção."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.text(0.5, 0.5, 'Clique num ponto para ver detalhes',
                         transform=self.ax_info.transAxes, ha='center', va='center',
                         fontsize=12, style='italic', color='gray')
        self.scatter.set_edgecolors('none')
        self.scatter.set_linewidths(0)
        self.fig.canvas.draw()
    
    def run(self):
        """Inicia a exploração interativa."""
        times = [s.time for s in self.pareto_front]
        co2s = [s.co2 for s in self.pareto_front]
        
        # Plot principal
        self.ax_plot.set_xlabel('Tempo de Viagem (min)', fontsize=12)
        self.ax_plot.set_ylabel('Emissões de CO₂ (g)', fontsize=12)
        self.ax_plot.set_title('Frente de Pareto - Clique para ver detalhes', fontsize=14, weight='bold')
        self.ax_plot.grid(True, alpha=0.3)
        
        self.scatter = self.ax_plot.scatter(times, co2s, s=80, alpha=0.7, 
                                           c='steelblue', picker=True, pickradius=5)
        
        # Marcar extremos
        extremes = self.results.get("extremes", {})
        if "best_time" in extremes:
            s = extremes["best_time"]
            self.ax_plot.scatter(s.time, s.co2, marker='*', s=300, c='green', 
                               edgecolors='black', linewidths=1, label='Melhor Tempo', zorder=5)
        if "best_co2" in extremes:
            s = extremes["best_co2"]
            self.ax_plot.scatter(s.time, s.co2, marker='*', s=300, c='red',
                               edgecolors='black', linewidths=1, label='Melhor CO₂', zorder=5)
        if "balanced" in extremes:
            s = extremes["balanced"]
            self.ax_plot.scatter(s.time, s.co2, marker='*', s=300, c='orange',
                               edgecolors='black', linewidths=1, label='Balanceada', zorder=5)
        
        self.ax_plot.legend(loc='upper right')
        
        # Painel de informações inicial
        self.ax_info.axis('off')
        self.ax_info.text(0.5, 0.5, 'Clique num ponto\npara ver detalhes',
                         transform=self.ax_info.transAxes, ha='center', va='center',
                         fontsize=14, style='italic', color='gray')
        
        # Botão de reset
        ax_button = plt.axes([0.85, 0.02, 0.1, 0.04])
        btn_reset = Button(ax_button, 'Reset')
        btn_reset.on_clicked(self._reset_selection)
        
        # Conectar eventos
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        
        plt.tight_layout()
        plt.show()


def main():
    print("\n=== Explorador Interativo da Frente de Pareto ===\n")
    print("Instruções:")
    print("  - Clique em qualquer ponto para ver os detalhes da solução")
    print("  - Estrelas marcam: Melhor Tempo (verde), Melhor CO₂ (vermelho), Balanceada (laranja)")
    print("  - Botão 'Reset' limpa a seleção\n")
    
    explorer = InteractiveParetoExplorer()
    explorer.run()


if __name__ == "__main__":
    main()
