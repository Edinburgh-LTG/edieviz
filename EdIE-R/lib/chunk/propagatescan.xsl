<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ng">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="w[@type~'_stroke$']">
        <xsl:apply-templates select="w[@type~'_stroke$']/@type"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="w[@type]/@type"/>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:if test="w[@sloc='deep']">
      <xsl:attribute name="loc-deep">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="w[@sloc='cortical']">
      <xsl:attribute name="loc-cortical">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="w[@stime='old']">
      <xsl:attribute name="time-old">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="w[@stime='recent']">
      <xsl:attribute name="time-recent">true</xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ag">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="w[@type~'_stroke$']">
        <xsl:apply-templates select="w[@type~'_stroke$']/@type"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="w[@type]/@type"/>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:if test="w[@sloc='deep']">
      <xsl:attribute name="loc-deep">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="w[@sloc='cortical']">
      <xsl:attribute name="loc-cortical">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="w[@stime='old']">
      <xsl:attribute name="time-old">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="w[@stime='recent']">
      <xsl:attribute name="time-recent">true</xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

