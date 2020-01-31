<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="meta"> 
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
    <header>
    <xsl:text>Report </xsl:text>
    <xsl:value-of select="../@id"/>
    <xsl:text>: </xsl:text>
    <xsl:value-of select="attr[@name='type']"/>
    <xsl:text> (</xsl:text>
    <xsl:value-of select="attr[@name='order_item']"/>
    <xsl:text>)</xsl:text>
    </header>
    <xsl:text>&#10;</xsl:text>
    <labels>
    <xsl:text>&#10;</xsl:text>
      <l id='1'>Ischaemic stroke, deep, recent</l><xsl:text>&#10;</xsl:text>
      <l id='2'>Ischaemic stroke, deep, old</l><xsl:text>&#10;</xsl:text>
      <l id='3'>Ischaemic stroke, cortical, recent</l><xsl:text>&#10;</xsl:text>
      <l id='4'>Ischaemic stroke, cortical, old</l><xsl:text>&#10;</xsl:text>
      <l id='5'>Ischaemic stroke, underspecified</l><xsl:text>&#10;</xsl:text>
      <l id='6'>Haemorrhagic stroke, deep, recent</l><xsl:text>&#10;</xsl:text>
      <l id='7'>Haemorrhagic stroke, deep, old</l><xsl:text>&#10;</xsl:text>
      <l id='8'>Haemorrhagic stroke, lobar, recent</l><xsl:text>&#10;</xsl:text>
      <l id='9'>Haemorrhagic stroke, lobar, old</l><xsl:text>&#10;</xsl:text>
      <l id='10'>Haemorrhagic stroke, underspecified</l><xsl:text>&#10;</xsl:text>
      <l id='11'>Stroke, underspecified</l><xsl:text>&#10;</xsl:text>
      <l id='12'>Tumour, meningioma</l><xsl:text>&#10;</xsl:text>
      <l id='13'>Tumour, metastasis</l><xsl:text>&#10;</xsl:text>
      <l id='14'>Tumour, glioma</l><xsl:text>&#10;</xsl:text>
      <l id='15'>Tumour, other</l><xsl:text>&#10;</xsl:text>
      <l id='16'>Small vessel disease</l><xsl:text>&#10;</xsl:text>
      <l id='17'>Atrophy</l><xsl:text>&#10;</xsl:text>
      <l id='18'>Subdural haematoma</l><xsl:text>&#10;</xsl:text>
      <l id='19'>Subarachnoid haemorrhage, aneurysmal</l><xsl:text>&#10;</xsl:text>
      <l id='20'>Subarachnoid haemorrhage, other</l><xsl:text>&#10;</xsl:text>
      <l id='21'>Microbleed, deep</l><xsl:text>&#10;</xsl:text>
      <l id='22'>Microbleed, lobar</l><xsl:text>&#10;</xsl:text>
      <l id='23'>Microbleed, underspecified</l><xsl:text>&#10;</xsl:text>
      <l id='24'>Haemorrhagic transformation</l><xsl:text>&#10;</xsl:text>
    </labels>
    <xsl:text>&#10;</xsl:text>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>
