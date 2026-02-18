import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

def _basic_explanation_for_row(row, prob):
    reasons = []
    if "Amount" in row.index:
        try:
            amount = float(row["Amount"])
            if amount >= 2000:
                reasons.append(f"High transaction amount: ${amount:.2f}")
            elif amount >= 500:
                reasons.append(f"Moderate amount: ${amount:.2f}")
            else:
                reasons.append(f"Low amount: ${amount:.2f}")
        except:
            pass
    
    reasons.append(f"Fraud probability: {prob:.2f}%")
    
    v_cols = [c for c in row.index if str(c).upper().startswith("V")]
    if v_cols:
        vals = []
        for c in v_cols:
            try:
                v = float(row[c])
                vals.append(v)
            except:
                continue
        if vals:
            abs_vals = np.abs(vals)
            big = (abs_vals > 3.0).sum()
            if big >= 3:
                reasons.append("Multiple unusual features detected")
            elif big > 0:
                reasons.append("Some unusual features")
    
    if "Time" in row.index:
        try:
            t = float(row["Time"])
            reasons.append(f"Transaction time: {t:.0f}s")
        except:
            pass
    
    return " | ".join(reasons)

def _plot_amount_prob_heatmap(amounts, probs):
    amounts = np.asarray(amounts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    
    valid_mask = (amounts >= 0) & (probs >= 0) & (amounts <= np.percentile(amounts, 99.5))
    amounts = amounts[valid_mask]
    probs = probs[valid_mask]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if len(amounts) == 0:
        ax.text(0.5, 0.5, "No valid data to display", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("Amount ($)")
        ax.set_ylabel("Fraud Probability (%)")
        ax.set_title("Amount vs Fraud Probability Heatmap")
    else:
        hb = ax.hexbin(
            amounts,
            probs,
            gridsize=30,
            mincnt=1,
            cmap="viridis",
            linewidths=0.1,
            edgecolors="gray"
        )
        
        max_amount = np.percentile(amounts, 95)
        if max_amount > 0:
            ax.set_xlim(0, max_amount * 1.1)
        
        max_prob = np.percentile(probs, 95)
        if max_prob > 0:
            ax.set_ylim(0, min(100, max_prob * 1.2))
        
        ax.set_xlabel("Transaction Amount ($)")
        ax.set_ylabel("Fraud Probability (%)")
        ax.set_title("Amount vs Fraud Probability Heatmap\n(Darker = More Transactions)")
        
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Number of Transactions")
        
        ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _plot_amount_prob_scatter(amounts, probs):
    amounts = np.asarray(amounts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    valid_mask = (amounts >= 0) & (probs >= 0)
    amounts_valid = amounts[valid_mask]
    probs_valid = probs[valid_mask]
    
    if len(amounts_valid) == 0:
        ax.text(0.5, 0.5, "No valid data to display", 
                ha='center', va='center', transform=ax.transAxes)
    else:
        scatter = ax.scatter(
            amounts_valid, 
            probs_valid, 
            c=probs_valid, 
            cmap="RdYlGn_r", 
            s=30, 
            alpha=0.6, 
            edgecolors="k", 
            linewidth=0.3,
            vmin=0, 
            vmax=100
        )
        
        ax.set_xlabel("Transaction Amount ($)")
        ax.set_ylabel("Fraud Probability (%)")
        ax.set_title("Transaction Risk Scatter Plot\n(Green = Lower Risk, Red = Higher Risk)")
        
        plt.colorbar(scatter, ax=ax, label="Fraud Probability (%)")
        
        ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def render_premium_csv_analytics(st_ref, original_df, results_df, probs):
    st_ref.success("Premium Mode Active - CSV Analytics")
    
    n_total = len(results_df)
    n_fraud = int((results_df["Status"] == "🚨 FRAUD").sum())
    avg_prob = float(np.nanmean(probs)) if len(probs) > 0 else 0.0
    
    st_ref.subheader("Summary")
    st_ref.markdown(
        f"- **Total Transactions:** {n_total:,}  \n"
        f"- **Fraud Cases Detected:** {n_fraud:,}  \n"
        f"- **Average Fraud Probability:** {avg_prob:.2f}%  \n"
    )
    
    if n_fraud > 0:
        st_ref.subheader("Top Fraud Cases")
        fraud_sorted = results_df.sort_values("Fraud_Probability", ascending=False).head(3)
        for _, r in fraud_sorted.iterrows():
            idx = int(r["ID"]) - 1
            if 0 <= idx < len(original_df):
                row = original_df.iloc[idx]
                expl = _basic_explanation_for_row(row, float(r["Fraud_Probability"]))
                st_ref.markdown(f"**Row {int(r['ID'])}** - {r['Fraud_Probability']:.2f}%")
                st_ref.markdown(f"*{expl}*")
                st_ref.markdown("---")
    
    st_ref.subheader("Transaction Distribution Heatmap")
    st_ref.markdown("Shows where most transactions cluster by amount and fraud probability.")
    img_bytes = _plot_amount_prob_heatmap(original_df.get("Amount", np.zeros(len(original_df))).values, probs)
    st_ref.image(img_bytes, use_container_width=True)
    
    st_ref.subheader("Individual Transaction Risk")
    st_ref.markdown("Each point is a transaction. Color indicates fraud risk level.")
    img2 = _plot_amount_prob_scatter(original_df.get("Amount", np.zeros(len(original_df))).values, probs)
    st_ref.image(img2, use_container_width=True)