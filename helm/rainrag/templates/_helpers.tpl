{{/*
Expand the name of the chart.
*/}}
{{- define "rainrag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "rainrag.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "rainrag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "rainrag.labels" -}}
helm.sh/chart: {{ include "rainrag.chart" . }}
{{ include "rainrag.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "rainrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rainrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Qdrant labels
*/}}
{{- define "rainrag.qdrant.labels" -}}
helm.sh/chart: {{ include "rainrag.chart" . }}
{{ include "rainrag.qdrant.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: qdrant
{{- end }}

{{/*
Qdrant selector labels
*/}}
{{- define "rainrag.qdrant.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rainrag.name" . }}-qdrant
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
vLLM labels
*/}}
{{- define "rainrag.vllm.labels" -}}
helm.sh/chart: {{ include "rainrag.chart" . }}
{{ include "rainrag.vllm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: vllm
{{- end }}

{{/*
vLLM selector labels
*/}}
{{- define "rainrag.vllm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rainrag.name" . }}-vllm
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
