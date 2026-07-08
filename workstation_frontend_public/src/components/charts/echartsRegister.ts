/**
 * Centralised ECharts component registration — keeps the bundle small by only
 * pulling in the charts / encoders we actually use across the cockpit.
 */
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart, CustomChart, ScatterChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkAreaComponent,
  LegendComponent,
  TitleComponent,
} from 'echarts/components'

let registered = false

export function ensureEchartsRegistered(): void {
  if (registered) return
  use([
    CanvasRenderer,
    LineChart,
    BarChart,
    CustomChart,
    ScatterChart,
    GridComponent,
    TooltipComponent,
    DataZoomComponent,
    MarkLineComponent,
    MarkAreaComponent,
    LegendComponent,
    TitleComponent,
  ])
  registered = true
}
