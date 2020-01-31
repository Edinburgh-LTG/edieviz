<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="l[@id='1']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @loc-deep='true' and @time-recent='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='2']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @loc-deep='true' and @time-old='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
<!-- when ischaemic stroke and deep but no time, then infer old -->
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @loc-deep='true') and not(@time-old or @time-recent)]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='3']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @loc-cortical='true' and @time-recent='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='4']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @loc-cortical='true' and @time-old='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='5']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @loc-deep='true') and not(@time-old or @time-recent)]">
      </xsl:when>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(ancestor::ng) and not(@neg) and (@type='ischaemic_stroke' and not(@*[name()~'^loc'] and @*[name()~'^time'])) and not(ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='ischaemic_stroke' and @*[name()~'^loc'] and @*[name()~'^time'])])]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='6']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='haemorrhagic_stroke' and @loc-deep='true' and @time-recent='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='7']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='haemorrhagic_stroke' and @loc-deep='true' and @time-old='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='8']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='haemorrhagic_stroke' and @loc-cortical='true' and @time-recent='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='9']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='haemorrhagic_stroke' and @loc-cortical='true' and @time-old='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='10']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[(not(ancestor::ng) and not(@neg) and @type='haemorrhagic_stroke' and not(@*[name()~'^loc'] and @*[name()~'^time'])) and not(ancestor::document//s[@proc='yes']//ng[not(@neg) and @type='haemorrhagic_stroke' and @*[name()~'^loc'] and @*[name()~'^time']])]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='11']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='stroke')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='12']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and @type='mening_tumour']">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='13']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and @type='metast_tumour']">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='14']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and @type='glioma_tumour']">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='15']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document[.//s[@proc='yes']//ng[not(@neg) and @type='tumour'] and not(.//ng[not(@neg) and @type~'_tumour$'])]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='16']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='small_vessel_disease')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='17']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='atrophy')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='18']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='subdural_haematoma')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='19']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes' and .//w[.~'^aneurysm']]//ng[not(@neg) and @type='subarachnoid_haemorrhage']">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='20']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes' and not(.//w[.~'^aneurysm'])]//ng[not(@neg) and @type='subarachnoid_haemorrhage']">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='21']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='microhaemorrhage' and @loc-deep='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='22']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='microhaemorrhage' and @loc-cortical='true')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='23']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(ancestor::ng) and not(@neg) and (@type='microhaemorrhage' and not(@*[name()~'^loc']))]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="l[@id='24']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::document//s[@proc='yes']//ng[not(@neg) and (@type='haemorrhagic_transformation')]">
        <xsl:attribute name="choice">pos</xsl:attribute>
      </xsl:when>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>
