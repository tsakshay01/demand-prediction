import { Doughnut } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    ArcElement,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function DonutChart({ data, title = 'Distribution' }) {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'right',
                labels: { color: '#94a3b8' }
            },
            title: {
                display: !!title,
                text: title,
                color: '#f8fafc'
            },
            tooltip: {
                backgroundColor: '#1e293b',
                borderColor: '#334155',
                borderWidth: 1
            }
        },
        cutout: '60%'
    };

    return <Doughnut data={data} options={options} />;
}
